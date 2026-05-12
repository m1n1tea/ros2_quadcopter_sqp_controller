from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "points_file",
                description="Absolute path to an Nx3 .npy trajectory file.",
            ),
            DeclareLaunchArgument("path_topic", default_value="/reference_path"),
            DeclareLaunchArgument("frame_id", default_value="map"),
            Node(
                package="quadrotor_acados",
                executable="path_publisher",
                name="path_publisher",
                output="screen",
                parameters=[
                    {
                        "points_file": LaunchConfiguration("points_file"),
                        "path_topic": LaunchConfiguration("path_topic"),
                        "frame_id": LaunchConfiguration("frame_id"),
                    }
                ],
            ),
        ]
    )
