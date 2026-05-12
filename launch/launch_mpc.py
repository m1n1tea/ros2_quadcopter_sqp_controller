from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("quadrotor_acados"))
    params_file = package_share / "config" / "x500.yaml"
    params_file_arg = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=str(params_file),
                description="Path to the parameters YAML file for px4_mpc_node",
            ),
            Node(
                package="quadrotor_acados",
                executable="px4_mpc_node",
                name="px4_mpc_node",
                output="screen",
                parameters=[params_file_arg],
            )
        ]
    )
