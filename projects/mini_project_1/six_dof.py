from math import *
import numpy as np
from typing import List, Tuple
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate, ScaraRobotTemplate, FiveDOFRobotTemplate, KinovaRobotTemplate
)


class KinovaRobot(KinovaRobotTemplate):
    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()
        th1, th2, th3, th4, th5, th6 = curr_joint_values

        offset = 0.50 * self.l1
        back   = 1.00 * self.l1
        fwd    = 0.50 * self.l1
        up     = 0.25 * self.l1
        j3_fwd = 0.25 * self.l1
        j5_bwd = 0.25 * self.l1
        drop   = 0.40 * self.l1

        c1, s1 = cos(th1), sin(th1)
        H1 = np.array([
            [ c1, 0, s1, 0],
            [  0, 1,  0, 0],
            [-s1, 0, c1, self.l1],
            [  0, 0,  0, 1]
        ])

        c2, s2 = cos(th2), sin(th2)
        H2 = np.array([
            [ c2, 0, s2, 0],
            [  0, 1,  0, 0],
            [-s2, 0, c2, self.l2],
            [  0, 0,  0, 1]
        ])

        d3 = self.l3 - offset - drop
        a3 = -self.l4 - back + j3_fwd
        c3, s3 = cos(th3), sin(th3)
        H3 = np.array([
            [c3, -s3, 0, a3 * c3],
            [s3,  c3, 0, a3 * s3],
            [ 0,   0, 1, d3],
            [ 0,   0, 0, 1]
        ])

        d4 = self.l5
        a4 = self.l4 + back - j3_fwd
        c4, s4 = cos(th4), sin(th4)
        H4 = np.array([
            [c4, -s4, 0, a4 * c4],
            [s4,  c4, 0, a4 * s4],
            [ 0,   0, 1, d4],
            [ 0,   0, 0, 1]
        ])

        d5 = -(5/3) * self.l5 + up
        a5 = self.l6 + fwd - j5_bwd
        c5, s5 = cos(th5), sin(th5)
        H5 = np.array([
            [c5, -s5, 0, a5 * c5],
            [s5,  c5, 0, a5 * s5],
            [ 0,   0, 1, d5],
            [ 0,   0, 0, 1]
        ])

        c6, s6 = cos(th6), sin(th6)
        H6 = np.array([
            [c6, -s6, 0, 0],
            [s6,  c6, 0, 0],
            [ 0,   0, 1, 0],
            [ 0,   0, 0, 1]
        ])

        H7 = np.eye(4)

        Hlist = [H1, H2, H3, H4, H5, H6, H7]

        H_ee = H1 @ H2 @ H3 @ H4 @ H5 @ H6 @ H7

        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = H_ee[:3, 3]

        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist


    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt=0.02):
        new_joint_values = joint_values.copy()

        if all(theta == 0.0 for theta in new_joint_values):
            new_joint_values = [
                new_joint_values[0] + 0.05,
                new_joint_values[1] + 0.10,
                new_joint_values[2] - 0.07,
                new_joint_values[3] + 0.04,
                new_joint_values[4] + 0.08,
                new_joint_values[5] + 0.03,
            ]

        vel = vel[:6] if len(vel) >= 6 else vel + [0] * (6 - len(vel))

        joint_vel = self.inverse_jacobian(new_joint_values) @ np.array(vel)

        vel_limits = self.joint_vel_limits
        if len(vel_limits) < 6:
            vel_limits = vel_limits + [vel_limits[-1]] * (6 - len(vel_limits))

        joint_vel = np.clip(joint_vel,
                            [limit[0] for limit in vel_limits],
                            [limit[1] for limit in vel_limits])

        for i in range(self.num_dof):
            new_joint_values[i] += dt * joint_vel[i]

        pos_limits = self.joint_limits
        if len(pos_limits) < 6:
            pos_limits = pos_limits + [pos_limits[-1]] * (6 - len(pos_limits))

        new_joint_values = np.clip(new_joint_values,
                                   [limit[0] for limit in pos_limits],
                                   [limit[1] for limit in pos_limits])

        return list(new_joint_values)


    def jacobian(self, joint_values: list):
        epsilon = 1e-6
        J = np.zeros((6, 6))

        ee_current, _ = self.calc_forward_kinematics(joint_values)
        current_pose = np.array([ee_current.x, ee_current.y, ee_current.z,
                                 ee_current.rotx, ee_current.roty, ee_current.rotz])

        for i in range(6):
            joint_values_plus = joint_values.copy()
            joint_values_plus[i] += epsilon

            ee_plus, _ = self.calc_forward_kinematics(joint_values_plus)
            pose_plus = np.array([ee_plus.x, ee_plus.y, ee_plus.z,
                                  ee_plus.rotx, ee_plus.roty, ee_plus.rotz])

            J[:, i] = (pose_plus - current_pose) / epsilon

        return J


    def inverse_jacobian(self, joint_values: list):
        J = self.jacobian(joint_values)
        damping = 0.05
        return J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))


if __name__ == "__main__":
    model = KinovaRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
