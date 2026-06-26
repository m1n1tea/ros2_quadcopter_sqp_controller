from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("quadrotor_acados"))
    default_params_file = package_share / "config" / "x500.yaml"
    params_file_arg = LaunchConfiguration("params_file")
    arguments = [
        DeclareLaunchArgument(
            "params_file",
            default_value=str(default_params_file),
            description="Path to the moving_target_publisher parameter YAML file",
        ),
        DeclareLaunchArgument(
            "attitude_setpoint_topic", default_value="/target/attitude_setpoint"
        ),
        DeclareLaunchArgument(
            "odometry_topic", default_value="/fmu/out/vehicle_odometry"
        ),
        DeclareLaunchArgument("x", default_value="5.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("z", default_value="-2.0"),
        DeclareLaunchArgument("vx", default_value="0.0"),
        DeclareLaunchArgument("vy", default_value="0.5"),
        DeclareLaunchArgument("vz", default_value="0.0"),
        DeclareLaunchArgument("publish_rate_hz", default_value="20.0"),
        DeclareLaunchArgument("log_interval_sec", default_value="1.0"),
        DeclareLaunchArgument("telemetry_log_interval_sec", default_value="0.1"),
        DeclareLaunchArgument("collision_radius_m", default_value="0.5"),
        DeclareLaunchArgument("direction_noise_std", default_value="0.0"),
        DeclareLaunchArgument("publish_distance", default_value="true"),
        DeclareLaunchArgument("distance_noise_std", default_value="0.0"),
        DeclareLaunchArgument("random_seed", default_value="0"),
        DeclareLaunchArgument("preferred_common_thrust", default_value="-1.0"),
        DeclareLaunchArgument("distance_thrust_gain", default_value="0.01"),
        DeclareLaunchArgument("min_common_thrust", default_value="0.0"),
        DeclareLaunchArgument("max_common_thrust", default_value="1.0"),
        DeclareLaunchArgument("max_tilt_deg", default_value="35.0"),
    ]

    return LaunchDescription(
        arguments
        + [
            Node(
                package="quadrotor_acados",
                executable="moving_target_publisher",
                name="moving_target_publisher",
                output="screen",
                parameters=[
                    params_file_arg,
                    {
                        "attitude_setpoint_topic": LaunchConfiguration(
                            "attitude_setpoint_topic"
                        ),
                        "odometry_topic": LaunchConfiguration("odometry_topic"),
                        "publish_distance": ParameterValue(
                            LaunchConfiguration("publish_distance"),
                            value_type=bool,
                        ),
                        "random_seed": ParameterValue(
                            LaunchConfiguration("random_seed"), value_type=int
                        ),
                        **{
                            name: ParameterValue(
                                LaunchConfiguration(name), value_type=float
                            )
                            for name in (
                                "x",
                                "y",
                                "z",
                                "vx",
                                "vy",
                                "vz",
                                "publish_rate_hz",
                                "log_interval_sec",
                                "telemetry_log_interval_sec",
                                "collision_radius_m",
                                "direction_noise_std",
                                "distance_noise_std",
                                "preferred_common_thrust",
                                "distance_thrust_gain",
                                "min_common_thrust",
                                "max_common_thrust",
                                "max_tilt_deg",
                            )
                        },
                    }
                ],
            )
        ]
    )
