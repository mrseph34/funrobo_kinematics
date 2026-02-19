from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate, ScaraRobotTemplate, FiveDOFRobotTemplate
)


class FiveDOFRobot(FiveDOFRobotTemplate):
    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()
        th1, th2, th3, th4, th5 = curr_joint_values

        c1, s1 = cos(th1), sin(th1)
        H0_1 = np.array([
            [c1,  0,  s1, 0],
            [s1,  0, -c1, 0],
            [0,   1,   0, self.l4],
            [0,   0,   0, 1]
        ])

        c2, s2 = cos(th2), sin(th2)
        H1_2 = np.array([
            [c2, -s2, 0, self.l1*c2],
            [s2,  c2, 0, self.l1*s2],
            [0,   0,  1, 0],
            [0,   0,  0, 1]
        ])

        c3, s3 = cos(th3), sin(th3)
        H2_3 = np.array([
            [c3, -s3, 0, self.l2*c3],
            [s3,  c3, 0, self.l2*s3],
            [0,   0,  1, 0],
            [0,   0,  0, 1]
        ])

        c4, s4 = cos(th4), sin(th4)
        H3_4 = np.array([
            [c4,  0,  s4, self.l3*c4],
            [s4,  0, -c4, self.l3*s4],
            [0,   1,   0, 0],
            [0,   0,   0, 1]
        ])

        c5, s5 = cos(th5), sin(th5)
        H4_5 = np.array([
            [c5, -s5, 0, 0],
            [s5,  c5, 0, 0],
            [0,   0,  1, 0],
            [0,   0,  0, 1]
        ])
        
        Hlist = [H0_1, H1_2, H2_3, H3_4, H4_5]
        H_ee = H0_1 @ H1_2 @ H2_3 @ H3_4 @ H4_5

        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = H_ee[:3, 3]
        
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist


    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt=0.02):
        new_joint_values = joint_values.copy()

        # Avoid singularity at zero configuration
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
        J = np.zeros((6, 5))
        
        ee_current, _ = self.calc_forward_kinematics(joint_values)
        current_pose = np.array([ee_current.x, ee_current.y, ee_current.z, 
                                ee_current.rotx, ee_current.roty, ee_current.rotz])
        
        for i in range(5):
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
    model = FiveDOFRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
