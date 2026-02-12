cmake_minimum_required(VERSION 3.8)
project(spinnaker_camera)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. FIND PACKAGES
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED)
find_package(nlohmann_json REQUIRED)      # Find the JSON library

include_directories(
  include
  /opt/spinnaker/include
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${image_transport_INCLUDE_DIRS}
  ${nlohmann_json_INCLUDE_DIRS}       # Add its include directory
)

# 2. ADD EXECUTABLE
add_executable(spinnaker_cam 
  src/spinnaker_cam.cpp 
  src/flir_controller.cpp
)

# 3. LINK LIBRARIES (Spinnaker SDK, OpenCV, JSON)
target_link_libraries(spinnaker_cam
  /opt/spinnaker/lib/libSpinnaker.so
  /opt/spinnaker/lib/libSpinnaker_C.so
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json        # Link against the JSON target
)

# 4. LINK ROS DEPENDENCIES
ament_target_dependencies(spinnaker_cam
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
)

install(TARGETS
  spinnaker_cam
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()