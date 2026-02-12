#ifndef FLIR_CONTROLLER_H
#define FLIR_CONTROLLER_H

#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// Spinnaker SDK
#include "SpinGenApi/SpinnakerGenApi.h"
#include "Spinnaker.h"

struct SensorStatus {
  int code;
  std::string msg;

  operator bool() const { return code == 0; }
};

struct SensorFrameView {
  const uint8_t *data;
  size_t size_bytes;
  uint32_t width;
  uint32_t height;
  uint64_t timestamp_ns;
  uint32_t frame_id;
};

using FrameCallback = std::function<void(const SensorFrameView &)>;

class FLIRCameraController {
public:
  FLIRCameraController();
  ~FLIRCameraController();

  SensorStatus set_device_id(const std::string &device_id);
  SensorStatus open();
  SensorStatus start();
  SensorStatus stop();
  SensorStatus close();

  SensorStatus set_property(const std::string &key,
                            const nlohmann::json &value);

  // NEW: Safely resets ROI/Binning and sets Width/Height to Sensor Max
  SensorStatus reset_to_max_resolution();

  void set_frame_callback(FrameCallback callback);

private:
  std::string device_id_;
  Spinnaker::SystemPtr system_;
  Spinnaker::CameraList cam_list_;
  Spinnaker::CameraPtr cam_;
  bool is_acquiring_;

  FrameCallback frame_callback_;

  class ImageEventHandlerImpl : public Spinnaker::ImageEventHandler {
  public:
    ImageEventHandlerImpl(FLIRCameraController *parent) : parent_(parent) {}
    void OnImageEvent(Spinnaker::ImagePtr image) override;

  private:
    FLIRCameraController *parent_;
  };

  std::shared_ptr<ImageEventHandlerImpl> image_handler_;

  friend class ImageEventHandlerImpl;
  void process_frame(Spinnaker::ImagePtr image);
};

#endif // FLIR_CONTROLLER_H