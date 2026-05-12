import casadi as cs
import numpy as np


def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
            ]
        )

    return cs.vertcat(
        cs.horzcat(1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
        cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)),
        cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)),
    )


def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)
    return cs.mtimes(rot_mat, v)


def quaternion_inverse(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    return cs.vertcat(w, -x, -y, -z)


def skew_symmetric(v):
    if isinstance(v, np.ndarray):
        return np.array(
            [[0, -v[0], -v[1], -v[2]], [v[0], 0, v[2], -v[1]], [v[1], -v[2], 0, v[0]], [v[2], v[1], -v[0], 0]]
        )

    return cs.vertcat(
        cs.horzcat(0, -v[0], -v[1], -v[2]),
        cs.horzcat(v[0], 0, v[2], -v[1]),
        cs.horzcat(v[1], -v[2], 0, v[0]),
        cs.horzcat(v[2], v[1], -v[0], 0),
    )


def dist(point1, point2):
    dxyz = point2[0:3] - point1[0:3]
    return np.sqrt(np.sum(dxyz**2))


def local_interpolate(point1, point2, preferred_dist):
    xyz1 = point1[0:3]
    xyz2 = point2[0:3]

    dxyz = xyz2 - xyz1
    segment_dist = np.sqrt(np.sum(dxyz**2))

    if preferred_dist > segment_dist:
        return point1

    t = preferred_dist / segment_dist
    point = np.zeros_like(point1)
    point[0:3] = xyz1 + t * dxyz
    return point


def transform_trajectory(traj, preferred_step):
    current_point = traj[0]
    passed_distance = 0.0
    new_traj = np.array([current_point])
    i = 0
    eps = 1e-9

    while i < len(traj):
        new_point = local_interpolate(current_point, traj[i], preferred_step - passed_distance)
        if dist(current_point, new_point) < eps:
            passed_distance += dist(current_point, traj[i])
            current_point = traj[i]
            i += 1
        else:
            new_traj = np.append(new_traj, np.array([new_point]), axis=0)
            current_point = new_point
            passed_distance = 0.0

    new_traj = np.append(new_traj, np.array([traj[-1]]), axis=0)
    return new_traj
