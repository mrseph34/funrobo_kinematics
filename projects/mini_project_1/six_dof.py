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

        H1 = ut.dh_to_matrix([th1,          self.l1,                          0,                          0])
        H2 = ut.dh_to_matrix([th2,          self.l2,                          0,                          0])
        H3 = ut.dh_to_matrix([th3,          self.l3 - offset - drop,         -self.l4 - back + j3_fwd,    0])
        H4 = ut.dh_to_matrix([th4,          self.l5,                          self.l4 + back - j3_fwd,    0])
        H5 = ut.dh_to_matrix([th5,         -(5/3) * self.l5 + up,             self.l6 + fwd - j5_bwd,     0])
        H6 = ut.dh_to_matrix([th6,          0,                                0,                          0])
        H7 = ut.dh_to_matrix([0,            0,                                0,                          0])

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
            new_joint_values = [theta + np.random.rand()*0.02 for theta in new_joint_values]

        vel = vel[:6] if len(vel) >= 6 else vel + [0]*(6-len(vel))

        joint_vel = self.inverse_jacobian(new_joint_values) @ vel

        joint_vel = np.clip(joint_vel,
                            [limit[0] for limit in self.joint_vel_limits],
                            [limit[1] for limit in self.joint_vel_limits])

        for i in range(self.num_dof):
            new_joint_values[i] += dt * joint_vel[i]

        new_joint_values = np.clip(new_joint_values,
                               [limit[0] for limit in self.joint_limits],
                               [limit[1] for limit in self.joint_limits])

        return new_joint_values


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
        return np.linalg.pinv(self.jacobian(joint_values))


if __name__ == "__main__":
    model = KinovaRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
