#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <SpinGenApi/SpinnakerGenApi.h>
#include <Spinnaker.h>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace cv;

class SpinnakerCamNode : public rclcpp::Node {
public:
  SpinnakerCamNode() : Node("spinnaker_camera") {
    pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw",
                                                           10);

    system_ = System::GetInstance();
    cam_list_ = system_->GetCameras();
    if (cam_list_.GetSize() == 0)
      throw std::runtime_error("No Spinnaker cameras detected");

    cam_ = cam_list_.GetByIndex(0);
    cam_->Init();
    cam_->AcquisitionMode.SetValue(AcquisitionMode_Continuous);
    cam_->BeginAcquisition();

    timer_ =
        this->create_wall_timer(std::chrono::milliseconds(1),
                                std::bind(&SpinnakerCamNode::grab_frame, this));

    RCLCPP_INFO(this->get_logger(), "Spinnaker camera node started");
  }

  ~SpinnakerCamNode() {
    cam_->EndAcquisition();
    cam_->DeInit();
    cam_list_.Clear();
    system_->ReleaseInstance();
  }

private:
  void grab_frame() {
    ImagePtr img = cam_->GetNextImage();
    if (img->IsIncomplete()) {
      img->Release();
      return;
    }

    ImageProcessor processor;
    ImagePtr converted = processor.Convert(img, PixelFormat_BGR8);

    cv::Mat frame(converted->GetHeight(), converted->GetWidth(), CV_8UC3,
                  converted->GetData());

    auto ros_msg =
        cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    ros_msg->header.frame_id = "camera";
    ros_msg->header.stamp = this->get_clock()->now();
    pub_->publish(*ros_msg);

    img->Release();
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  SystemPtr system_;
  CameraList cam_list_;
  CameraPtr cam_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SpinnakerCamNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
