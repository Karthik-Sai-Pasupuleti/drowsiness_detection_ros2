cmake_minimum_required(VERSION 3.8)
project(spinnaker_camera)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  /opt/spinnaker/include
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
)

add_executable(spinnaker_cam src/spinnaker_cam.cpp)

target_link_libraries(spinnaker_cam
  /opt/spinnaker/lib/libSpinnaker.so
  /opt/spinnaker/lib/libSpinnaker_C.so
  ${OpenCV_LIBS}
)

ament_target_dependencies(spinnaker_cam
  rclcpp
  sensor_msgs
  cv_bridge
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
