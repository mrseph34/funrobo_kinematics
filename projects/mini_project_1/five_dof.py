from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate, ScaraRobotTemplate, FiveDOFRobotTemplate
)


MAX_JOINT_DELTA_DEG = 3.0
MAX_JOINT_DELTA_RAD = np.deg2rad(MAX_JOINT_DELTA_DEG)


class FiveDOFRobot(FiveDOFRobotTemplate):
    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()

        if not radians:
            curr_joint_values = [np.deg2rad(theta) for theta in curr_joint_values]

        for i, theta in enumerate(curr_joint_values):
            curr_joint_values[i] = np.clip(theta, self.joint_limits[i][0], self.joint_limits[i][1])

        DH = np.zeros((self.num_dof, 4))
        DH[0] = [curr_joint_values[0], self.l1, 0, -np.pi/2]
        DH[1] = [curr_joint_values[1] - np.pi/2, 0, self.l2, np.pi]
        DH[2] = [curr_joint_values[2], 0, self.l3, np.pi]
        DH[3] = [curr_joint_values[3] + np.pi/2, 0, 0, np.pi/2]
        DH[4] = [curr_joint_values[4], self.l4 + self.l5, 0, 0]

        Hlist = [ut.dh_to_matrix(dh) for dh in DH]

        H_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            H_cumulative.append(H_cumulative[-1] @ Hlist[i])

        H_ee = H_cumulative[-1]

        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]

        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist


    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt=0.02):
        new_joint_values = joint_values.copy()

        if abs(new_joint_values[0]) < 0.05 and abs(new_joint_values[4]) < 0.05:
            new_joint_values[0] += 0.05
            new_joint_values[4] += 0.05

        vel = vel[:3] if len(vel) >= 3 else vel + [0] * (3 - len(vel))

        J = self.jacobian(new_joint_values)
        JT = J.T
        J_inv = JT @ np.linalg.inv(J @ JT + (.025**2) * np.eye(3))

        joint_vel = J_inv @ vel
        joint_vel = np.clip(joint_vel,
                            [limit[0] for limit in self.joint_vel_limits],
                            [limit[1] for limit in self.joint_vel_limits])

        for i in range(self.num_dof):
            delta = np.clip(dt * joint_vel[i], -MAX_JOINT_DELTA_RAD, MAX_JOINT_DELTA_RAD)
            new_joint_values[i] += delta

        new_joint_values = np.clip(new_joint_values,
                                   [limit[0] for limit in self.joint_limits],
                                   [limit[1] for limit in self.joint_limits])

        return new_joint_values


    def jacobian(self, joint_values: list):
        _, Hlist = self.calc_forward_kinematics(joint_values)

        H_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            H_cumulative.append(H_cumulative[-1] @ Hlist[i])

        O0 = np.array([0, 0, 0, 1])
        J = np.zeros((3, self.num_dof))

        for i in range(self.num_dof):
            r = (H_cumulative[-1] @ O0 - H_cumulative[i] @ O0)[:3]
            z = H_cumulative[i][:3, :3] @ np.array([0, 0, 1])
            J[:, i] = np.cross(z, r)

        J[np.abs(J) < 1e-10] = 0.0
        self._last_jacobian = J
        return J


    def inverse_jacobian(self, joint_values: list):
        J = self.jacobian(joint_values)
        JT = J.T
        return JT @ np.linalg.inv(J @ JT + (.025**2) * np.eye(3))


if __name__ == "__main__":
    from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
    model = FiveDOFRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()