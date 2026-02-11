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
        
        # Link lengths in meters (from Hiwonder documentation)
        # Following the diagram: 3 main arm segments
        self.l1 = 0.106  # First arm segment (shoulder to elbow)
        self.l2 = 0.106  # Second arm segment (elbow to wrist)  
        self.l3 = 0.166  # Third arm segment (wrist to end) - combined l3 + wrist_length
        self.base_height = 0.106  # Base height


    def calc_forward_kinematics(self, joint_values: list, radians=True):
        """
        Calculate forward kinematics for the 5DOF Hiwonder arm.
        
        5 joints total:
        - Joint 1 (th1): Base rotation (around Z)
        - Joint 2 (th2): Shoulder pitch 
        - Joint 3 (th3): Elbow pitch
        - Joint 4 (th4): Wrist pitch  
        - Joint 5 (th5): Wrist roll
        
        The arm structure from the diagram:
        x = l₁cos(θ₂) + l₂cos(θ₂+θ₃) + l₃cos(θ₂+θ₃+θ₄)
        (then rotated by θ₁ around Z, and rolled by θ₅)
        
        Args:
            joint_values (list): [th1, th2, th3, th4, th5]
            
        Returns:
            ee: EndEffector object with position and orientation
            Hlist: List of transformation matrices for each joint
        """
        curr_joint_values = joint_values.copy()
        
        # Extract joint angles
        th1, th2, th3, th4, th5 = curr_joint_values
        l1, l2, l3 = self.l1, self.l2, self.l3

        # ===== JOINT 1: Base rotation around Z-axis =====
        # This rotates the entire arm in the horizontal plane
        H0_1 = np.array([
            [cos(th1), -sin(th1), 0, 0],
            [sin(th1),  cos(th1), 0, 0],
            [0,         0,        1, self.base_height],
            [0,         0,        0, 1]
        ])

        # ===== JOINT 2: Shoulder pitch =====
        # Rotate by th2, then translate by l1 along local X
        c2, s2 = cos(th2), sin(th2)
        H1_2 = np.array([
            [c2,  0, s2, l1],
            [0,   1, 0,  0],
            [-s2, 0, c2, 0],
            [0,   0, 0,  1]
        ])

        # ===== JOINT 3: Elbow pitch =====
        # Rotate by th3, then translate by l2 along local X
        c3, s3 = cos(th3), sin(th3)
        H2_3 = np.array([
            [c3,  0, s3, l2],
            [0,   1, 0,  0],
            [-s3, 0, c3, 0],
            [0,   0, 0,  1]
        ])

        # ===== JOINT 4: Wrist pitch =====
        # Rotate by th4, then translate by l3 along local X
        c4, s4 = cos(th4), sin(th4)
        H3_4 = np.array([
            [c4,  0, s4, l3],
            [0,   1, 0,  0],
            [-s4, 0, c4, 0],
            [0,   0, 0,  1]
        ])

        # ===== JOINT 5: Wrist roll around X-axis =====
        # Just rotation, no translation (end effector is at the origin of this frame)
        c5, s5 = cos(th5), sin(th5)
        H4_5 = np.array([
            [1, 0,   0,   0],
            [0, c5, -s5,  0],
            [0, s5,  c5,  0],
            [0, 0,   0,   1]
        ])
        
        # Store all transformation matrices (5 matrices for 5 DOF)
        Hlist = [H0_1, H1_2, H2_3, H3_4, H4_5]

        # Calculate final end effector transformation
        H_ee = H0_1 @ H1_2 @ H2_3 @ H3_4 @ H4_5

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
        
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist


    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt=0.02):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            joint_values (list): Current joint angles [th1, th2, th3, th4, th5]
            vel (list): The velocity vector for the end effector [vx, vy, vz, wx, wy, wz]
            dt (float): Time step for integration
            
        Returns:
            new_joint_values (list): Updated joint angles
        """
        new_joint_values = joint_values.copy()

        # Move robot slightly out of zeros singularity
        if all(theta == 0.0 for theta in new_joint_values):
            new_joint_values = [theta + np.random.rand()*0.02 for theta in new_joint_values]
        
        # Calculate joint velocities using the inverse Jacobian
        # For 5DOF, we use the first 5 components of velocity (3 linear + 2 angular)
        vel = vel[:5] if len(vel) >= 5 else vel + [0]*(5-len(vel))
        joint_vel = self.inverse_jacobian(new_joint_values) @ vel
        
        # Clip joint velocities to limits
        joint_vel = np.clip(joint_vel, 
                            [limit[0] for limit in self.joint_vel_limits], 
                            [limit[1] for limit in self.joint_vel_limits]
                        )

        # Update the joint angles based on the velocity
        for i in range(self.num_dof):
            new_joint_values[i] += dt * joint_vel[i]

        # Ensure joint angles stay within limits
        new_joint_values = np.clip(new_joint_values, 
                               [limit[0] for limit in self.joint_limits], 
                               [limit[1] for limit in self.joint_limits]
                            )
        
        return new_joint_values


    def jacobian(self, joint_values: list):
        """
        Returns the Jacobian matrix for the 5DOF robot.
        
        Uses the geometric approach based on the diagram's formulation.
        Following: x = l₁cos(θ₂) + l₂cos(θ₂+θ₃) + l₃cos(θ₂+θ₃+θ₄)

        Args:
            joint_values (list): The joint angles [th1, th2, th3, th4, th5]

        Returns:
            np.ndarray: The Jacobian matrix (5x5)
        """
        th1, th2, th3, th4, th5 = joint_values
        l1, l2, l3 = self.l1, self.l2, self.l3
        
        # Calculate cumulative angles (as shown in the diagram)
        th2_only = th2
        th23 = th2 + th3
        th234 = th2 + th3 + th4
        
        # Base rotation effect
        # The planar arm reach in the base-rotated frame
        arm_reach_x = l1*cos(th2_only) + l2*cos(th23) + l3*cos(th234)
        arm_reach_z = l1*sin(th2_only) + l2*sin(th23) + l3*sin(th234)
        
        # Jacobian matrix (5x5)
        J = np.zeros((5, 5))
        
        # Column 0: Effect of th1 (base rotation) - rotates everything in XY plane
        J[0, 0] = -sin(th1) * arm_reach_x  # dx/dth1
        J[1, 0] =  cos(th1) * arm_reach_x  # dy/dth1
        J[2, 0] = 0  # dz/dth1 (base rotation doesn't change height)
        
        # Column 1: Effect of th2 (shoulder pitch)
        J[0, 1] = cos(th1) * (-l1*sin(th2_only) - l2*sin(th23) - l3*sin(th234))
        J[1, 1] = sin(th1) * (-l1*sin(th2_only) - l2*sin(th23) - l3*sin(th234))
        J[2, 1] = l1*cos(th2_only) + l2*cos(th23) + l3*cos(th234)
        
        # Column 2: Effect of th3 (elbow pitch)
        J[0, 2] = cos(th1) * (-l2*sin(th23) - l3*sin(th234))
        J[1, 2] = sin(th1) * (-l2*sin(th23) - l3*sin(th234))
        J[2, 2] = l2*cos(th23) + l3*cos(th234)
        
        # Column 3: Effect of th4 (wrist pitch)
        J[0, 3] = cos(th1) * (-l3*sin(th234))
        J[1, 3] = sin(th1) * (-l3*sin(th234))
        J[2, 3] = l3*cos(th234)
        
        # Column 4: Effect of th5 (wrist roll) - minimal position effect, mainly rotational
        J[0, 4] = 0
        J[1, 4] = 0
        J[2, 4] = 0
        J[3, 4] = 1  # Angular velocity around X
        J[4, 4] = 0
        
        return J
    

    def inverse_jacobian(self, joint_values: list):
        """
        Returns the inverse (pseudo-inverse) of the Jacobian matrix.
        
        Uses Moore-Penrose pseudo-inverse for redundant/non-square matrices.

        Args:
            joint_values (list): Current joint angles
            
        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        return np.linalg.pinv(self.jacobian(joint_values))


if __name__ == "__main__":
    model = FiveDOFHiwonder()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()