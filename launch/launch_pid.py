from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="quadrotor_acados",
                executable="px4_pid_node",
                name="px4_pid_node",
                output="screen",
            ),
        ]
    )
