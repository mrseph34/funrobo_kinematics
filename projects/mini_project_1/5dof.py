from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate, ScaraRobotTemplate, FiveDOFRobotTemplate
)


class FiveDOFHiwonder(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()
        
        # Link lengths in meters
        self.l1 = 0.106
        self.l2 = 0.106
        self.l3 = 0.166
        self.base_height = 0.106


    def calc_forward_kinematics(self, joint_values: list, radians=True):
        curr_joint_values = joint_values.copy()
        
        th1, th2, th3, th4, th5 = curr_joint_values
        l1, l2, l3 = self.l1, self.l2, self.l3

        # Base rotation around Z-axis
        H0_1 = np.array([
            [cos(th1), -sin(th1), 0, 0],
            [sin(th1),  cos(th1), 0, 0],
            [0,         0,        1, self.base_height],
            [0,         0,        0, 1]
        ])

        # Shoulder pitch
        c2, s2 = cos(th2), sin(th2)
        H1_2 = np.array([
            [c2,  0, s2, l1],
            [0,   1, 0,  0],
            [-s2, 0, c2, 0],
            [0,   0, 0,  1]
        ])

        # Elbow pitch
        c3, s3 = cos(th3), sin(th3)
        H2_3 = np.array([
            [c3,  0, s3, l2],
            [0,   1, 0,  0],
            [-s3, 0, c3, 0],
            [0,   0, 0,  1]
        ])

        # Wrist pitch
        c4, s4 = cos(th4), sin(th4)
        H3_4 = np.array([
            [c4,  0, s4, l3],
            [0,   1, 0,  0],
            [-s4, 0, c4, 0],
            [0,   0, 0,  1]
        ])

        # Wrist roll around X-axis
        c5, s5 = cos(th5), sin(th5)
        H4_5 = np.array([
            [1, 0,   0,   0],
            [0, c5, -s5,  0],
            [0, s5,  c5,  0],
            [0, 0,   0,   1]
        ])
        
        Hlist = [H0_1, H1_2, H2_3, H3_4, H4_5]

        # Calculate end effector transformation
        H_ee = H0_1 @ H1_2 @ H2_3 @ H3_4 @ H4_5

        # Set end effector position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract RPY from rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist


    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt=0.02):
        new_joint_values = joint_values.copy()

        # Move robot slightly out of zeros singularity
        if all(theta == 0.0 for theta in new_joint_values):
            new_joint_values = [theta + np.random.rand()*0.02 for theta in new_joint_values]
        
        # Calculate joint velocities using inverse Jacobian
        vel = vel[:5] if len(vel) >= 5 else vel + [0]*(5-len(vel))
        joint_vel = self.inverse_jacobian(new_joint_values) @ vel
        
        # Clip joint velocities to limits
        joint_vel = np.clip(joint_vel, 
                            [limit[0] for limit in self.joint_vel_limits], 
                            [limit[1] for limit in self.joint_vel_limits]
                        )

        # Update joint angles
        for i in range(self.num_dof):
            new_joint_values[i] += dt * joint_vel[i]

        # Ensure joint angles stay within limits
        new_joint_values = np.clip(new_joint_values, 
                               [limit[0] for limit in self.joint_limits], 
                               [limit[1] for limit in self.joint_limits]
                            )
        
        return new_joint_values


    def jacobian(self, joint_values: list):
        th1, th2, th3, th4, th5 = joint_values
        l1, l2, l3 = self.l1, self.l2, self.l3
        
        # Calculate cumulative angles
        th2_only = th2
        th23 = th2 + th3
        th234 = th2 + th3 + th4
        
        # Planar arm reach in base-rotated frame
        arm_reach_x = l1*cos(th2_only) + l2*cos(th23) + l3*cos(th234)
        arm_reach_z = l1*sin(th2_only) + l2*sin(th23) + l3*sin(th234)
        
        J = np.zeros((5, 5))
        
        # Base rotation effect
        J[0, 0] = -sin(th1) * arm_reach_x
        J[1, 0] =  cos(th1) * arm_reach_x
        J[2, 0] = 0
        
        # Shoulder pitch effect
        J[0, 1] = cos(th1) * (-l1*sin(th2_only) - l2*sin(th23) - l3*sin(th234))
        J[1, 1] = sin(th1) * (-l1*sin(th2_only) - l2*sin(th23) - l3*sin(th234))
        J[2, 1] = l1*cos(th2_only) + l2*cos(th23) + l3*cos(th234)
        
        # Elbow pitch effect
        J[0, 2] = cos(th1) * (-l2*sin(th23) - l3*sin(th234))
        J[1, 2] = sin(th1) * (-l2*sin(th23) - l3*sin(th234))
        J[2, 2] = l2*cos(th23) + l3*cos(th234)
        
        # Wrist pitch effect
        J[0, 3] = cos(th1) * (-l3*sin(th234))
        J[1, 3] = sin(th1) * (-l3*sin(th234))
        J[2, 3] = l3*cos(th234)
        
        # Wrist roll effect
        J[0, 4] = 0
        J[1, 4] = 0
        J[2, 4] = 0
        J[3, 4] = 1
        J[4, 4] = 0
        
        return J
    

    def inverse_jacobian(self, joint_values: list):
        return np.linalg.pinv(self.jacobian(joint_values))


if __name__ == "__main__":
    model = FiveDOFHiwonder()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()