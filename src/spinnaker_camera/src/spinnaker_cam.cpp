#include <memory>

#include <string>

#include <vector>

#include <chrono>
 
// ROS 2 Core

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>

#include <sensor_msgs/image_encodings.hpp>

#include <image_transport/image_transport.hpp>

#include <cv_bridge/cv_bridge.h>
 
// OpenCV

#include <opencv2/opencv.hpp>
 
// Your provided header

#include "flir_controller.h"
 
using namespace std::chrono_literals;
 
class FlirRosNode : public rclcpp::Node {

public:

    FlirRosNode() : Node("flir_camera_node") {

        // 1. Declare Parameters

        declare_parameters();
 
        // 2. Initialize Camera Controller

        try {

            camera_controller_ = std::make_shared<FLIRCameraController>();

            std::string device_id;

            this->get_parameter("device_id", device_id);
 
            if (device_id.empty()) {

                RCLCPP_ERROR(this->get_logger(), "Parameter 'device_id' is required.");

                throw std::runtime_error("Device ID missing");

            }
 
            RCLCPP_INFO(this->get_logger(), "Looking for device: %s", device_id.c_str());

            // Set ID and Open

            SensorStatus status = camera_controller_->set_device_id(device_id);

            if (!status) throw std::runtime_error(status.msg);

            status = camera_controller_->open();

            if (!status) throw std::runtime_error("Failed to open camera: " + status.msg);
 
            // 3. Apply Configuration Parameters

            apply_configuration();
 
            // 4. Setup Publisher

            // We use image_transport for efficient publishing (supports compressed plugins)

            pub_ = image_transport::create_camera_publisher(this, "image_raw");
 
            // 5. Register Callback and Start

            camera_controller_->set_frame_callback(

                std::bind(&FlirRosNode::publish_frame, this, std::placeholders::_1)

            );
 
            status = camera_controller_->start();

            if (!status) throw std::runtime_error("Failed to start acquisition: " + status.msg);
 
            RCLCPP_INFO(this->get_logger(), "FLIR Camera Node started successfully.");
 
        } catch (const std::exception &e) {

            RCLCPP_FATAL(this->get_logger(), "Initialization failed: %s", e.what());

            // Allow the node to crash so the launch system knows it failed

            rclcpp::shutdown(); 

        }

    }
 
    ~FlirRosNode() {

        if (camera_controller_) {

            camera_controller_->stop();

            camera_controller_->close();

        }

    }
 
private:

    std::shared_ptr<FLIRCameraController> camera_controller_;

    image_transport::CameraPublisher pub_;

    sensor_msgs::msg::CameraInfo camera_info_msg_; 

    void declare_parameters() {

        // Name, Default Value, Description

        this->declare_parameter("device_id", ""); // Mandatory

        this->declare_parameter("PixelFormat", "BGR8"); // New Parameter

        this->declare_parameter("GainAuto", "Off");

        this->declare_parameter("Gain", 10.0);

        this->declare_parameter("ExposureAuto", "Off");

        this->declare_parameter("ExposureTime", 10000.0);

        this->declare_parameter("BalanceWhiteAuto", "Continuous");

        // Note: GammaEnable is usually a bool in GenICam, but user requested Double.

        // We will assume this double value is the Gamma *Value*, and implies enabling Gamma.

        this->declare_parameter("GammaEnable", 0.45); 

    }
 
    void apply_configuration() {

        RCLCPP_INFO(this->get_logger(), "Applying camera configuration...");
 
        // Helper lambda to set properties and log errors

        auto set_prop = [&](const std::string& key, const nlohmann::json& val) {

            SensorStatus s = camera_controller_->set_property(key, val);

            if (!s) {

                RCLCPP_WARN(this->get_logger(), "Failed to set %s: %s", key.c_str(), s.msg.c_str());

            } else {

                RCLCPP_INFO(this->get_logger(), "Set %s to %s", key.c_str(), val.dump().c_str());

            }

        };
 
        // --- Pixel Format ---

        // Must be set before starting acquisition to ensure buffer sizes are correct

        std::string px_fmt = this->get_parameter("PixelFormat").as_string();

        set_prop("PixelFormat", px_fmt);
 
        // --- Exposure & Gain ---

        set_prop("ExposureAuto", this->get_parameter("ExposureAuto").as_string());

        // Only set manual time if Auto is Off

        if (this->get_parameter("ExposureAuto").as_string() == "Off") {

             set_prop("ExposureTime", this->get_parameter("ExposureTime").as_double());

        }
 
        set_prop("GainAuto", this->get_parameter("GainAuto").as_string());

        if (this->get_parameter("GainAuto").as_string() == "Off") {

            set_prop("Gain", this->get_parameter("Gain").as_double());

        }
 
        // --- White Balance ---

        set_prop("BalanceWhiteAuto", this->get_parameter("BalanceWhiteAuto").as_string());
 
        // --- Gamma ---

        // Requirement: "GammaEnable: double : default 0.45"

        // Mapping: Enable Gamma boolean, and set the Gamma float value.

        double gamma_val = this->get_parameter("GammaEnable").as_double();

        set_prop("GammaEnable", true);

        set_prop("Gamma", gamma_val);

    }
 
    void publish_frame(const SensorFrameView& view) {

        if (!view.data || view.size_bytes == 0) return;
 
        // Determine Encoding based on buffer size and resolution

        // Bpp = Size / (Width * Height)

        size_t pixels = view.width * view.height;

        size_t bpp = (pixels > 0) ? (view.size_bytes / pixels) : 0;

        std::string encoding;

        int cv_type;
 
        // Logic to detect encoding based on bytes-per-pixel

        if (bpp == 1) {

            encoding = sensor_msgs::image_encodings::MONO8;

            cv_type = CV_8UC1;

        } else if (bpp == 2) {

            // Likely Mono16 or Bayer16 (Spinnaker sends raw bytes)

            // CV_16UC1 handles 16-bit grayscale

            encoding = sensor_msgs::image_encodings::MONO16;

            cv_type = CV_16UC1;

        } else if (bpp == 3) {

            // BGR8 or RGB8Packed. The controller defaults to BGR8 (OpenCV Standard)

            encoding = sensor_msgs::image_encodings::BGR8;

            cv_type = CV_8UC3;

        } else {

            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 

                "Unsupported bytes-per-pixel: %zu (Width: %u, Height: %u, Size: %zu). Skipping frame.", 

                bpp, view.width, view.height, view.size_bytes);

            return;

        }
 
        // Create OpenCV Mat (Zero-copy wrapper around raw buffer)

        // const_cast is necessary because cv::Mat constructor takes void*, but doesn't modify it unless drawn upon

        cv::Mat frame(view.height, view.width, cv_type, const_cast<void*>(view.data));
 
        // Create Header

        std_msgs::msg::Header header;

        header.stamp = this->now(); // Using ROS time for compatibility.

        header.frame_id = "camera_optical_frame";
 
        // Convert to ROS Message

        // Note: For Bayer formats (e.g., BayerBG8), Spinnaker sends raw 1-channel data. 

        // If 'PixelFormat' is BayerBG8, bpp is 1, so it is published as MONO8/BAYER_BGGR8. 

        // If debayering is needed, an image_proc node should be used downstream.

        sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(header, encoding, frame).toImageMsg();
 
        // Publish

        // Camera Info (Dummy for now, in a real node you'd use camera_info_manager)

        camera_info_msg_.header = header;

        camera_info_msg_.width = view.width;

        camera_info_msg_.height = view.height;

        pub_.publish(msg, std::make_shared<sensor_msgs::msg::CameraInfo>(camera_info_msg_));

    }

};
 
int main(int argc, char **argv) {

    rclcpp::init(argc, argv);

    auto node = std::make_shared<FlirRosNode>();

    rclcpp::spin(node);

    rclcpp::shutdown();

    return 0;

}
 
