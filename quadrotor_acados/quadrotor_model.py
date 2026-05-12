from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass
class QuadrotorParams:
    mass: float
    min_thrust: float
    max_thrust: float
    J: np.ndarray
    x_f: np.ndarray
    y_f: np.ndarray
    z_l_tau: np.ndarray
    max_input_value: float = field(init=False, default=1.0)
    min_input_value: float = field(init=False, default=0.0)



def load_params(mpc_node) -> QuadrotorParams:

    mass = mpc_node.get_parameter("uav.parameters.uav_mass").value

    min_speed = mpc_node.get_parameter("uav.parameters.min_rotor_speed").value
    max_speed = mpc_node.get_parameter("uav.parameters.max_rotor_speed").value
    thrust_constant = mpc_node.get_parameter("uav.parameters.thrust_constant").value
    min_thrust = (min_speed**2) * thrust_constant
    max_thrust = (max_speed**2) * thrust_constant
    J = np.array([mpc_node.get_parameter("uav.parameters.inertia.xx").value, mpc_node.get_parameter("uav.parameters.inertia.yy").value, mpc_node.get_parameter("uav.parameters.inertia.zz").value])
    x_f = np.array(mpc_node.get_parameter("uav.parameters.rotor_x").value)
    y_f = np.array(mpc_node.get_parameter("uav.parameters.rotor_y").value)
    z_l_tau = np.array(mpc_node.get_parameter("uav.parameters.rotor_direction").value) * mpc_node.get_parameter("uav.parameters.moment_constant").value
    return QuadrotorParams(
        mass=mass,
        min_thrust=min_thrust,
        max_thrust=max_thrust,
        J=J,
        x_f=x_f,
        y_f=y_f,
        z_l_tau=z_l_tau
    )
