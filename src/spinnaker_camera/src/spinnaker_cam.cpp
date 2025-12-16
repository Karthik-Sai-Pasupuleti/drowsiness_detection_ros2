#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// Assuming this header defines FLIRCameraController, SensorStatus, etc.
#include "flir_controller.h"

// Assuming you're using nlohmann/json for the set_property lambda
#include <nlohmann/json.hpp> // <-- Add this if you don't have it already

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
            // Using "camera/image_raw" to match your original topic name
            pub_ = image_transport::create_camera_publisher(this, "camera/image_raw");

            // 5. Register Callback and Start
            camera_controller_->set_frame_callback(
                std::bind(&FlirRosNode::publish_frame, this, std::placeholders::_1)
            );

            status = camera_controller_->start();
            if (!status) throw std::runtime_error("Failed to start acquisition: " + status.msg);

            RCLCPP_INFO(this->get_logger(), "FLIR Camera Node started successfully.");

        } catch (const std::exception &e) {
            RCLCPP_FATAL(this->get_logger(), "Initialization failed: %s", e.what());
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

    // NEW LOGIC: Updated to include Height and Width
    void declare_parameters() {
        // --- Required/Critical Parameters ---
        this->declare_parameter("device_id", ""); // Mandatory
        this->declare_parameter("PixelFormat", "BGR8");
        
        // --- NEW: Camera Dimensions ---
        this->declare_parameter("image_width", 0);  // 0 means use maximum/default
        this->declare_parameter("image_height", 0); // 0 means use maximum/default

        // --- Exposure & Gain ---
        this->declare_parameter("GainAuto", "Off");
        this->declare_parameter("Gain", 10.0);
        this->declare_parameter("ExposureAuto", "Off");
        this->declare_parameter("ExposureTime", 10000.0);
        this->declare_parameter("BalanceWhiteAuto", "Continuous");
        this->declare_parameter("GammaEnable", 0.45);
    }

    // NEW LOGIC: Updated to apply Height and Width
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

        // --- 1. Dimension/Format Settings (Must be first) ---
        
        // The GenICam features are typically "Width" and "Height", not "image_width"
        int width = this->get_parameter("image_width").as_int();
        int height = this->get_parameter("image_height").as_int();
        
        if (width > 0) {
            set_prop("Width", width);
        }
        if (height > 0) {
            set_prop("Height", height);
        }

        std::string px_fmt = this->get_parameter("PixelFormat").as_string();
        set_prop("PixelFormat", px_fmt);

        // --- 2. Exposure & Gain ---
        set_prop("ExposureAuto", this->get_parameter("ExposureAuto").as_string());
        if (this->get_parameter("ExposureAuto").as_string() == "Off") {
            set_prop("ExposureTime", this->get_parameter("ExposureTime").as_double());
        }

        set_prop("GainAuto", this->get_parameter("GainAuto").as_string());
        if (this->get_parameter("GainAuto").as_string() == "Off") {
            set_prop("Gain", this->get_parameter("Gain").as_double());
        }

        // --- 3. Color/Misc Settings ---
        set_prop("BalanceWhiteAuto", this->get_parameter("BalanceWhiteAuto").as_string());
        
        double gamma_val = this->get_parameter("GammaEnable").as_double();
        set_prop("GammaEnable", true);
        set_prop("Gamma", gamma_val);
        
        // Finalize Camera Info with the dimensions that were successfully set
        // In a complete driver, you would read the actual set dimensions back from the camera
        // For simplicity, we'll initialize the camera info here:
        camera_info_msg_.width = width; // Will be 0 if not set, but better practice is reading from camera
        camera_info_msg_.height = height; // Same here

        RCLCPP_INFO(this->get_logger(), "Configuration applied. Image size request: %dx%d", width, height);
    }

    void publish_frame(const SensorFrameView& view) {
        if (!view.data || view.size_bytes == 0) return;

        // Determine Encoding based on buffer size and resolution
        size_t pixels = view.width * view.height;
        size_t bpp = (pixels > 0) ? (view.size_bytes / pixels) : 0;

        std::string encoding;
        int cv_type;

        if (bpp == 1) {
            encoding = sensor_msgs::image_encodings::MONO8;
            cv_type = CV_8UC1;
        } else if (bpp == 2) {
            encoding = sensor_msgs::image_encodings::MONO16;
            cv_type = CV_16UC1;
        } else if (bpp == 3) {
            encoding = sensor_msgs::image_encodings::BGR8;
            cv_type = CV_8UC3;
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                "Unsupported bytes-per-pixel: %zu (Width: %u, Height: %u, Size: %zu). Skipping frame.", 
                bpp, view.width, view.height, view.size_bytes);
            return;
        }

        // Create OpenCV Mat (Zero-copy wrapper around raw buffer)
        cv::Mat frame(view.height, view.width, cv_type, const_cast<void*>(view.data));

        // Create Header
        std_msgs::msg::Header header;
        header.stamp = this->now();
        header.frame_id = "camera_optical_frame";

        // Convert to ROS Message
        sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(header, encoding, frame).toImageMsg();

        // Publish (Update Camera Info with actual frame dimensions for consistency)
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
