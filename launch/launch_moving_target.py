from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument("target_topic", default_value="/target/odometry"),
        DeclareLaunchArgument(
            "odometry_topic", default_value="/fmu/out/vehicle_odometry"
        ),
        DeclareLaunchArgument("frame_id", default_value="ned"),
        DeclareLaunchArgument("x", default_value="5.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        DeclareLaunchArgument("z", default_value="-2.0"),
        DeclareLaunchArgument("vx", default_value="0.0"),
        DeclareLaunchArgument("vy", default_value="0.5"),
        DeclareLaunchArgument("vz", default_value="0.0"),
        DeclareLaunchArgument("publish_rate_hz", default_value="20.0"),
        DeclareLaunchArgument("log_interval_sec", default_value="1.0"),
        DeclareLaunchArgument("direction_noise_std", default_value="0.0"),
        DeclareLaunchArgument("publish_distance", default_value="true"),
        DeclareLaunchArgument("distance_noise_std", default_value="0.0"),
        DeclareLaunchArgument("random_seed", default_value="0"),
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
                    {
                        "target_topic": LaunchConfiguration("target_topic"),
                        "odometry_topic": LaunchConfiguration("odometry_topic"),
                        "frame_id": LaunchConfiguration("frame_id"),
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
                                "direction_noise_std",
                                "distance_noise_std",
                            )
                        },
                    }
                ],
            )
        ]
    )
