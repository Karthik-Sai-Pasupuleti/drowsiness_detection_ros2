from setuptools import find_packages, setup
import os
from glob import glob

package_name = "drowsiness_detection_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    include_package_data=True,
    package_data={
        "drowsiness_detection_pkg.carla": ["wheel_config.ini"],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (
            "share/drowsiness_detection/bot/configs",
            [
                "drowsiness_detection/bot/configs/prompt.toml",
                "drowsiness_detection/bot/configs/schema.json",
            ],
        ),
        (
            os.path.join("share", package_name, "templates"),
            glob("drowsiness_detection/labelling/templates/*.html"),
        ),
        (
            os.path.join("share", package_name, "static"),
            glob("drowsiness_detection/labelling/static/**/*", recursive=True),
        ),
    ],
    install_requires=[
        "setuptools",
    ],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@todo.todo",
    description="Driver assistance package for drowsiness detection",
    license="TODO: License declaration",
    entry_points={
        "console_scripts": [
            "mediapipe_node = drowsiness_detection.camera.camera_mediapipe_node:main",
            "carla_node = drowsiness_detection.carla.carla_manual_control:main",
            "driver_assistance_node = drowsiness_detection.main:main",
            "drowsiness_gui = drowsiness_detection.labelling.app_v2:main",
            "steering_vibration_node = drowsiness_detection.bot.controls.steering_node:main",
            "speaker_node = drowsiness_detection.bot.controls.speaker_node:main",
            "fan_node = drowsiness_detection.bot.controls.fan_node:main",
            "steering_wheel_ffb_node = drowsiness_detection.bot.controls.steering_wheel_autocenter:main",
        ],
    },
)
