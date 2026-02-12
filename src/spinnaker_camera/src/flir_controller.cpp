#include "flir_controller.h"
#include <iostream>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;

void FLIRCameraController::ImageEventHandlerImpl::OnImageEvent(ImagePtr image) {
  if (parent_) {
    parent_->process_frame(image);
  }
}

FLIRCameraController::FLIRCameraController() : is_acquiring_(false) {
  system_ = System::GetInstance();
}

FLIRCameraController::~FLIRCameraController() {
  stop();
  close();
  if (system_) {
    system_->ReleaseInstance();
  }
}

SensorStatus FLIRCameraController::set_device_id(const std::string &device_id) {
  device_id_ = device_id;
  return {0, "Device ID set"};
}

SensorStatus FLIRCameraController::open() {
  try {
    cam_list_ = system_->GetCameras();
    if (cam_list_.GetSize() == 0) {
      return {-1, "No cameras found"};
    }

    cam_ = cam_list_.GetBySerial(device_id_);
    if (!cam_) {
      return {-1, "Camera with serial " + device_id_ + " not found"};
    }

    cam_->Init();

    image_handler_ = std::make_shared<ImageEventHandlerImpl>(this);
    cam_->RegisterEventHandler(*image_handler_);

    return {0, "Camera Opened"};
  } catch (const Spinnaker::Exception &e) {
    return {-1, std::string("Spinnaker Error: ") + e.what()};
  }
}

SensorStatus FLIRCameraController::close() {
  try {
    if (cam_) {
      if (is_acquiring_)
        stop();

      if (image_handler_) {
        cam_->UnregisterEventHandler(*image_handler_);
        image_handler_ = nullptr;
      }

      cam_->DeInit();
      cam_ = nullptr;
    }
    cam_list_.Clear();
    return {0, "Closed"};
  } catch (const Spinnaker::Exception &e) {
    return {-1, e.what()};
  }
}

SensorStatus FLIRCameraController::start() {
  try {
    if (!cam_)
      return {-1, "Camera not initialized"};
    if (is_acquiring_)
      return {0, "Already acquiring"};

    INodeMap &nodeMap = cam_->GetNodeMap();
    CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
    if (IsAvailable(ptrAcquisitionMode) && IsWritable(ptrAcquisitionMode)) {
      CEnumEntryPtr ptrContinuous =
          ptrAcquisitionMode->GetEntryByName("Continuous");
      if (IsAvailable(ptrContinuous) && IsReadable(ptrContinuous)) {
        ptrAcquisitionMode->SetIntValue(ptrContinuous->GetValue());
      }
    }

    cam_->BeginAcquisition();
    is_acquiring_ = true;
    return {0, "Started"};
  } catch (const Spinnaker::Exception &e) {
    return {-1, e.what()};
  }
}

SensorStatus FLIRCameraController::stop() {
  try {
    if (cam_ && is_acquiring_) {
      cam_->EndAcquisition();
      is_acquiring_ = false;
    }
    return {0, "Stopped"};
  } catch (const Spinnaker::Exception &e) {
    return {-1, e.what()};
  }
}

void FLIRCameraController::set_frame_callback(FrameCallback callback) {
  frame_callback_ = callback;
}

void FLIRCameraController::process_frame(ImagePtr image) {
  if (image->IsIncomplete()) {
    return;
  }

  if (frame_callback_) {
    SensorFrameView view;
    view.width = image->GetWidth();
    view.height = image->GetHeight();
    view.data = (const uint8_t *)image->GetData();
    view.size_bytes = image->GetBufferSize();
    view.frame_id = image->GetFrameID();
    view.timestamp_ns = image->GetTimeStamp();

    frame_callback_(view);
  }
}

// NEW: Robust Resolution Reset
SensorStatus FLIRCameraController::reset_to_max_resolution() {
  if (!cam_)
    return {-1, "Camera not open"};

  try {
    INodeMap &nodeMap = cam_->GetNodeMap();

    // 1. Reset Binning to 1 (Horizontal and Vertical)
    CIntegerPtr ptrBinH = nodeMap.GetNode("BinningHorizontal");
    if (IsAvailable(ptrBinH) && IsWritable(ptrBinH))
      ptrBinH->SetValue(1);

    CIntegerPtr ptrBinV = nodeMap.GetNode("BinningVertical");
    if (IsAvailable(ptrBinV) && IsWritable(ptrBinV))
      ptrBinV->SetValue(1);

    // 2. Reset Offsets to 0
    CIntegerPtr ptrOffsetX = nodeMap.GetNode("OffsetX");
    if (IsAvailable(ptrOffsetX) && IsWritable(ptrOffsetX))
      ptrOffsetX->SetValue(0);

    CIntegerPtr ptrOffsetY = nodeMap.GetNode("OffsetY");
    if (IsAvailable(ptrOffsetY) && IsWritable(ptrOffsetY))
      ptrOffsetY->SetValue(0);

    // 3. Set Width/Height to their Max (Read Max first!)
    CIntegerPtr ptrWidth = nodeMap.GetNode("Width");
    if (IsAvailable(ptrWidth) && IsWritable(ptrWidth)) {
      ptrWidth->SetValue(ptrWidth->GetMax());
    }

    CIntegerPtr ptrHeight = nodeMap.GetNode("Height");
    if (IsAvailable(ptrHeight) && IsWritable(ptrHeight)) {
      ptrHeight->SetValue(ptrHeight->GetMax());
    }

    return {0, "Reset to Max Resolution"};
  } catch (const Spinnaker::Exception &e) {
    return {-1, std::string("Reset Error: ") + e.what()};
  }
}

SensorStatus FLIRCameraController::set_property(const std::string &key,
                                                const nlohmann::json &value) {
  if (!cam_)
    return {-1, "Camera not open"};

  try {
    INodeMap &nodeMap = cam_->GetNodeMap();

    if (value.is_number_float()) {
      CFloatPtr ptr = nodeMap.GetNode(key.c_str());
      if (IsAvailable(ptr) && IsWritable(ptr)) {
        ptr->SetValue(value.get<double>());
        return {0, "OK"};
      }
    }

    if (value.is_number_integer()) {
      CIntegerPtr ptr = nodeMap.GetNode(key.c_str());
      if (IsAvailable(ptr) && IsWritable(ptr)) {
        ptr->SetValue(value.get<int64_t>());
        return {0, "OK"};
      }
    }

    if (value.is_boolean()) {
      CBooleanPtr ptr = nodeMap.GetNode(key.c_str());
      if (IsAvailable(ptr) && IsWritable(ptr)) {
        ptr->SetValue(value.get<bool>());
        return {0, "OK"};
      }
    }

    if (value.is_string()) {
      std::string val_str = value.get<std::string>();
      CEnumerationPtr ptrEnum = nodeMap.GetNode(key.c_str());
      if (IsAvailable(ptrEnum) && IsWritable(ptrEnum)) {
        CEnumEntryPtr ptrEntry = ptrEnum->GetEntryByName(val_str.c_str());
        if (IsAvailable(ptrEntry) && IsReadable(ptrEntry)) {
          ptrEnum->SetIntValue(ptrEntry->GetValue());
          return {0, "OK"};
        }
      }
      CStringPtr ptrString = nodeMap.GetNode(key.c_str());
      if (IsAvailable(ptrString) && IsWritable(ptrString)) {
        ptrString->SetValue(val_str.c_str());
        return {0, "OK"};
      }
    }
    return {-1, "Property '" + key + "' not found or not writable"};
  } catch (const Spinnaker::Exception &e) {
    return {-1, e.what()};
  }
}