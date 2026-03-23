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

        th1, th2, th3, th4, th5 = curr_joint_values

        t1 = th1
        t2 = th2 # - np.pi/2
        t3 = th3
        t4 = th4 + np.pi/2
        t5 = th5

        # DH: [theta, d, a, alpha]
        # H = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)

        c1, s1 = cos(t1), sin(t1)
        H0_1 = np.array([
            [ c1, -s1,  0,  0         ],
            [ s1,  c1,  0,  0         ],
            [  0,   0,  1,  self.l1   ],
            [  0,   0,  0,  1         ]
        ]) @ np.array([
            [ 1,  0,  0,  0 ],
            [ 0,  0,  1,  0 ],
            [ 0, -1,  0,  0 ],
            [ 0,  0,  0,  1 ]
        ])

        c2, s2 = cos(t2), sin(t2)
        H1_2 = np.array([
            [ c2, -s2,  0,  self.l2*c2 ],
            [ s2,  c2,  0,  self.l2*s2 ],
            [  0,   0,  1,  0          ],
            [  0,   0,  0,  1          ]
        ]) @ np.array([
            [ 1,  0,  0,  0 ],
            [ 0, -1,  0,  0 ],
            [ 0,  0, -1,  0 ],
            [ 0,  0,  0,  1 ]
        ])

        c3, s3 = cos(t3), sin(t3)
        H2_3 = np.array([
            [ c3, -s3,  0,  self.l3*c3 ],
            [ s3,  c3,  0,  self.l3*s3 ],
            [  0,   0,  1,  0          ],
            [  0,   0,  0,  1          ]
        ]) @ np.array([
            [ 1,  0,  0,  0 ],
            [ 0, -1,  0,  0 ],
            [ 0,  0, -1,  0 ],
            [ 0,  0,  0,  1 ]
        ])

        c4, s4 = cos(t4), sin(t4)
        H3_4 = np.array([
            [ c4, -s4,  0,  0 ],
            [ s4,  c4,  0,  0 ],
            [  0,   0,  1,  0 ],
            [  0,   0,  0,  1 ]
        ]) @ np.array([
            [ 1,  0,  0,  0 ],
            [ 0,  0, -1,  0 ],
            [ 0,  1,  0,  0 ],
            [ 0,  0,  0,  1 ]
        ])

        c5, s5 = cos(t5), sin(t5)
        H4_5 = np.array([
            [ c5, -s5,  0,  0                  ],
            [ s5,  c5,  0,  0                  ],
            [  0,   0,  1,  self.l4 + self.l5  ],
            [  0,   0,  0,  1                  ]
        ])

        Hlist = [H0_1, H1_2, H2_3, H3_4, H4_5]

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

        vel = vel[:3] if len(vel) >= 3 else vel + [0] * (3 - len(vel))
        vel = np.array(vel, dtype=float)

        J = self.jacobian(new_joint_values)

        manipulability = np.sqrt(max(0.0, np.linalg.det(J @ J.T)))
        if manipulability < .01:
            lam = .05 * (1.0 - manipulability / .01)
        else:
            lam = 0.0

        J_dls = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(J.shape[0]))

        joint_vel = J_dls @ vel
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
        epsilon = 1e-5
        J = np.zeros((3, self.num_dof))

        ee_current, _ = self.calc_forward_kinematics(joint_values)
        f0 = np.array([ee_current.x, ee_current.y, ee_current.z])

        for i in range(self.num_dof):
            joint_values_plus = joint_values.copy()
            joint_values_plus[i] += epsilon

            ee_plus, _ = self.calc_forward_kinematics(joint_values_plus)
            f1 = np.array([ee_plus.x, ee_plus.y, ee_plus.z])

            J[:, i] = (f1 - f0) / epsilon

        J[np.abs(J) < 1e-10] = 0.0
        self._last_jacobian = J
        return J


    def inverse_jacobian(self, joint_values: list):
        return np.linalg.pinv(self.jacobian(joint_values))


if __name__ == "__main__":
    from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
    model = FiveDOFRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()