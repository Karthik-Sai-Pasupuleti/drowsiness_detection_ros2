from setuptools import find_packages, setup
import os
from glob import glob

package_name = "drowsiness_detection_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    include_package_data=True,  # important to include package_data
    package_data={
        "drowsiness_detection_pkg.carla": ["wheel_config.ini"],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (
            "share/drowsiness_detection_pkg/bot/configs",
            [
                "drowsiness_detection_pkg/bot/configs/prompt.toml",
                "drowsiness_detection_pkg/bot/configs/schema.json",
            ],
        ),
    ],
    install_requires=[
        "setuptools",
        "carla_msgs",
        "pygame",
    ],  # include pygame since carla_manual_control uses it
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@todo.todo",
    description="Driver assistance package for drowsiness detection",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_node = drowsiness_detection_pkg.camera.camera_node:main",
            "mediapipe_node = drowsiness_detection_pkg.camera.mediapipe_node:main",
            "carla_node = drowsiness_detection_pkg.carla.carla_manual_control:main",
            "driver_assistance_node = drowsiness_detection_pkg.driver_assistance_node:main",
            "drowsiness_gui = drowsiness_detection_pkg.data_labelling_gui:main",
            "steering_vibration_node = drowsiness_detection_pkg.bot.controls.steering_node:main",
            "speaker_node = drowsiness_detection_pkg.bot.controls.speaker_node:main",
            "fan_node = drowsiness_detection_pkg.bot.controls.fan_node:main",
            "steering_wheel_ffb_node = drowsiness_detection_pkg.bot.controls.steering_wheel_autocenter:main",
        ],
    },
)
