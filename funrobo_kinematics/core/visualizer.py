"""
GUI-based robot manipulator visualizer for the FunRobo kinematics library.

This module provides two main classes:

- Visualizer:
    A Tkinter GUI for interacting with a robot simulation:
    - Forward kinematics (set joint values)
    - Inverse kinematics (set end-effector pose)
    - Velocity kinematics (keyboard-controlled end-effector velocity)
    - Simple waypoint loading and trajectory demo hooks

- RobotSim:
    A lightweight simulation/visualization wrapper around a robot "model" object.
    It uses Matplotlib (embedded in Tkinter) to render joint positions, links, and
    coordinate frames.

Expected robot model interface
------------------------------
RobotSim expects `robot_model` to provide at least:
    - num_dof (int)
    - joint_values (list[float])
    - ee (EndEffector)
    - points (list[np.ndarray])
    - EE_axes (np.ndarray shape (3, 3) or similar)
    - calc_forward_kinematics(joint_values, radians=True/False) -> (EndEffector, list[np.ndarray])
    - calc_inverse_kinematics(ee, joint_values, soln=0) -> list[float]
    - calc_numerical_ik(ee, joint_values, tol=..., ilimit=...) -> list[float]
    - calc_velocity_kinematics(joint_values, vel, dt=...) -> list[float]
    - calc_robot_points(joint_values, Hlist, radians=True/False) -> None

Conventions:
- Lengths are meters.
- Angles are radians internally; the GUI often accepts degrees for convenience.
- FK returns a list of **individual link transforms** (one per joint), not cumulative transforms.

Notes for teaching:
- This module is intended for interactive exploration, not real-time control.
- Some features (trajectory generation) assume external classes/functions exist.

"""

import math
import time
from typing import List, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import yaml
from pynput import keyboard

from funrobo_kinematics.core.utils import EndEffector, wraptopi


class Visualizer:
    """
    Tkinter GUI for visualizing and controlling a robot manipulator.

    The GUI embeds a Matplotlib figure and provides panels for:
    - Forward kinematics: set joint values via entries or sliders (degrees shown)
    - Inverse kinematics: set a desired end-effector pose and solve IK
    - Velocity kinematics: toggle keyboard-controlled velocity motion
    - Waypoints / trajectory demos (optional hooks)

    Args:
        robot: A RobotSim-like object that owns a Matplotlib Figure and exposes
            methods such as `update_plot(...)` and `move_velocity(...)`.

    Attributes:
        robot: Robot simulation/visualization wrapper.
        root: Tkinter root window owned by this Visualizer.
        canvas: Tkinter-embedded Matplotlib canvas.
        vk_status: Whether velocity-control mode is active.
        v: Current commanded Cartesian velocity [vx, vy, vz] used in VK mode.
        listener: Background keyboard listener used for VK mode.
    """

    def __init__(self, robot) -> None:
        """
         Initialize the Visualizer and build the GUI.

        This creates the Tkinter root window, starts the keyboard listener
        for velocity control, and builds the GUI layout.

        Args:
            robot: A robot simulation/visualization object (e.g., RobotSim).
        """
        self.robot = robot

        # Create tk root
        self.root = tk.Tk()
        title = f"Robot Manipulator Visualization ({self.robot.num_joints}-DOF)"
        self.root.title(title)

        print(
            f"\nInitialized the Robot Manipulator Visualization for the "
            f"{self.robot.num_joints}-DOF robot for Kinematics Analysis\n"
        )
        
        # Variables for velocity kinematics
        self.vk_status = False
        self.v = [0.0, 0.0, 0.0]  # [vx, vy, vz]
        self.vk_status_font = ('black')

        # Keyboard listener for velocity kinematics
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

        # Build the GUI
        self._build_layout()

    
    def _build_layout(self) -> None:
        """
        Build the main GUI layout.

        Creates:
        - Left: control panel frame (Tk widgets)
        - Right: plot frame embedding the robot's Matplotlib figure
        """
        # Create the control frame for the GUI
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.grid(row=0, column=0, padx=15, pady=15)
        
        # Create the plot frame for the GUI
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10)

        # Embed the robot's figure into the Tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.robot.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0)

        # Set up the kinematics panel
        self.set_kinematics_panel()

        
    def set_kinematics_panel(self) -> None:
        """
        Create the kinematics control panel widgets.

        Panels include:
        - Forward Kinematics: joint entries + sliders + reset
        - Inverse Kinematics: end-effector pose entries + solve buttons
        - Velocity Kinematics: activate/deactivate buttons
        - Trajectory Generation: waypoint upload + demo generation buttons

        Returns:
            None. Widgets are created and stored as instance attributes.
        """
        # ------------------------------------------------------------------------------------------------
        # Forward position kinematics
        # ------------------------------------------------------------------------------------------------
        self.joint_values = []
        row_number = 0

        # Add title for the forward kinematics entry field
        self.fk_entry_title = ttk.Label(self.control_frame, text="Forward Kinematics:", font=("Arial", 13, "bold"))
        self.fk_entry_title.grid(column=0, row=row_number, columnspan=2, pady=(0, 10))
        row_number += 1

        # Create joint entry fields and labels
        self.joint_button = []
        for i in range(self.robot.num_joints):
            joint_label = ttk.Label(self.control_frame, text=f"theta {i+1} (deg or m):")
            joint_label.grid(column=0, row=row_number, sticky=tk.W)
            joint_value = ttk.Entry(self.control_frame)
            joint_value.insert(0, "0")
            joint_value.grid(column=1, row=row_number)
            self.joint_button.append(joint_value)
            row_number += 1

        # Create the Move button
        self.fk_move_button = ttk.Button(self.control_frame, text="Move", command=self.joints_from_button)
        self.fk_move_button.grid(column=0, row=row_number, columnspan=2, pady=5)
        row_number += 1

        # Create the joint slider field and labels
        self.joint_scales = []
        for i in range(self.robot.num_joints):
            joint_label = ttk.Label(self.control_frame, text=f"theta {i+1} (deg or m):")
            joint_label.grid(column=0, row=row_number, sticky=tk.W)

            joint_value = tk.DoubleVar()
            slider = ttk.Scale(
                self.control_frame, 
                from_=-180, 
                to=180, 
                variable=joint_value, 
                command=self.joints_from_sliders
            )
            slider.grid(column=1, row=row_number)
            row_number += 1
            self.joint_scales.append(joint_value)

        # Create the Reset button
        self.fk_reset_button = ttk.Button(
            self.control_frame, 
            text="Reset", 
            command=self.reset_joints
        )
        self.fk_reset_button.grid(column=0, row=row_number, columnspan=2, pady=5)
        row_number += 3

        # ------------------------------------------------------------------------------------------------
        # Inverse position kinematics
        # ------------------------------------------------------------------------------------------------
        self.ik_entry_title = ttk.Label(
            self.control_frame, 
            text="Inverse Kinematics:", 
            font=("Arial", 13, "bold")
        )
        self.ik_entry_title.grid(column=0, row=row_number, columnspan=2, pady=(0, 10))
        row_number += 1

        # Create end-effector pose field and labels
        self.pose_button = []
        pose_labels = ['X(m)', 'Y(m)', 'Z(m)', 'RotX(rad)', 'RotY(rad)', 'RotZ(rad)']
        for i in range(len(pose_labels)):
            position_label = ttk.Label(self.control_frame, text=pose_labels[i] + ":")
            position_label.grid(column=0, row=row_number, sticky=tk.W)
            position_value = ttk.Entry(self.control_frame)
            position_value.insert(0, "0")
            position_value.grid(column=1, row=row_number)
            row_number += 1
            self.pose_button.append(position_value)

        # Create buttons for inverse kinematics solutions
        self.ik1_move_button = ttk.Button(
            self.control_frame, 
            text="Solve 1", 
            command=self.solve_IK1
        )
        self.ik1_move_button.grid(column=0, row=row_number, columnspan=1, pady=2)

        self.ik2_move_button = ttk.Button(
            self.control_frame, 
            text="Solve 2", 
            command=self.solve_IK2
        )
        self.ik2_move_button.grid(column=1, row=row_number, columnspan=1, pady=2)

        self.ik3_move_button = ttk.Button(
            self.control_frame, 
            text="Num Solve", 
            command=self.numerical_solve
        )
        self.ik3_move_button.grid(column=2, row=row_number, columnspan=1, pady=2)

        self.ik_set_pose_button = ttk.Button(
            self.control_frame,
            text="Set Pose",
            command=self.load_current_pose
        )
        self.ik_set_pose_button.grid(column=1, row=row_number+1, columnspan=1, pady=1)
        row_number += 5
        

        # ------------------------------------------------------------------------------------------------
        # Velocity kinematics
        # ------------------------------------------------------------------------------------------------
        self.vk_entry_title = ttk.Label(self.control_frame, text="Velocity Kinematics:", font=("Arial", 13, "bold"))
        self.vk_entry_title.grid(column=0, row=row_number, columnspan=2, pady=(0, 10))
        row_number += 1

        self.vk_activate_button = ttk.Button(self.control_frame, text="Activate VK", command=self.activate_VK)
        self.vk_activate_button.grid(column=0, row=row_number, columnspan=1, pady=2)

        self.vk_deactivate_button = ttk.Button(self.control_frame, text="Deactivate VK", command=self.deactivate_VK)
        self.vk_deactivate_button.grid(column=1, row=row_number, columnspan=1, pady=2)
        row_number += 3

        # ------------------------------------------------------------------------------------------------
        # Trajectory generation
        # ------------------------------------------------------------------------------------------------
        self.tg_entry_title = ttk.Label(self.control_frame, text="Trajectory Generation:", font=("Arial", 13, "bold"))
        self.tg_entry_title.grid(column=0, row=row_number, columnspan=2, pady=(0, 10))
        row_number += 1

        self.tg_generate_button = ttk.Button(self.control_frame, text="Upload Waypoints", command=self.update_waypoints)
        self.tg_generate_button.grid(column=0, row=row_number, columnspan=1, pady=2)

        self.tg_follow_task_button = ttk.Button(self.control_frame, text="Generate (Task-space)", command=self.generate_traj_task_space)
        self.tg_follow_task_button.grid(column=1, row=row_number, columnspan=1, pady=2)

        self.tg_follow_joint_button = ttk.Button(self.control_frame, text="Generate (Joint-space)", command=self.generate_traj_joint_space)
        self.tg_follow_joint_button.grid(column=2, row=row_number, columnspan=1, pady=2)
        row_number += 1


    def joints_from_sliders(self, val) -> None:
        """
        Updates the forward kinematics based on the joint angles set by the sliders.
        """
        joint_values = [float(th.get()) for th in self.joint_scales]
        self.update_FK(joint_values)


    def joints_from_button(self) -> None:
        """
        Updates the forward kinematics based on the joint angles entered in the input fields.
        """
        joint_values = [float(th.get()) for th in self.joint_button]
        self.update_FK(joint_values)


    def reset_joints(self) -> None:
        """
        Resets all joint angles to 0 and updates the forward kinematics.
        """
        joint_values = [0.0] * self.robot.num_joints
        self.robot.reset_ee_trajectory()

        # --- Reset sliders ---
        for var in self.joint_scales:
            var.set(0.0)

        # --- Reset entry boxes ---
        for entry in self.joint_button:
            entry.delete(0, tk.END)
            entry.insert(0, "0")

        self.update_FK(joint_values)


    def get_ee_from_input(self) -> EndEffector:
        """
        Read the end-effector pose from the IK input fields.

        Returns:
            EndEffector: End-effector pose populated from the GUI entries.
        """
        EE = EndEffector()
        EE.x = float(self.pose_button[0].get())
        EE.y = float(self.pose_button[1].get())
        EE.z = float(self.pose_button[2].get())
        EE.rotx = float(self.pose_button[3].get())
        EE.roty = float(self.pose_button[4].get())
        EE.rotz = float(self.pose_button[5].get())
        return EE
    

    def solve_IK1(self) -> None:
        """
        Solves the inverse kinematics for a given end-effector pose using the first solution.
        """
        self.update_IK(pose=self.get_ee_from_input(), soln=0)


    def solve_IK2(self) -> None:
        """
        Solves the inverse kinematics for a given end-effector pose using the second solution.
        """
        self.update_IK(pose=self.get_ee_from_input(), soln=1)


    def numerical_solve(self) -> None:
        """
        Solves the inverse kinematics for a given end-effector pose using a numerical method.
        """
        self.update_IK(pose=self.get_ee_from_input(), soln=1, numerical=True)


    def update_FK(self, joint_values: List[float], display_traj: bool = False) -> None:
        """
        Update the visualization using forward kinematics.

        The GUI accepts joint values in degrees for convenience; they are converted
        to radians before calling the robot simulation.

        Args:
            joint_values: Joint values in degrees (revolute joints) or meters (prismatic joints).
            display_traj: If True, appends the current EE pose to the trajectory trace.

        Returns:
            None. Updates the plot in-place.

        Raises:
            ValueError: If any joint input cannot be parsed as a float.
        """
        if display_traj:
            self.robot.update_ee_trajectory()
        
        try:
            self.robot.update_plot(joint_values=np.deg2rad(joint_values))
            self.canvas.draw()
            self.canvas.flush_events()
        except ValueError:
            tk.messagebox.showerror("Input Error", "Please enter valid numbers")


    def update_IK(
        self, pose: EndEffector, soln: int = 0, numerical: bool = False, display_traj: bool = False
    ) -> None:
        """
        Updates the inverse kinematics plot based on the given end-effector pose.

        Args:
            pose: Desired end-effector pose.
            soln: Solution branch index used by analytical IK. Defaults to 0.
            numerical: If True, use numerical IK. Defaults to False.
            display_traj: If True, appends the current EE pose to the trajectory trace.
        """
        if display_traj:
            self.robot.update_ee_trajectory()
        
        if numerical:
            self.robot.update_plot(pose=pose, soln=soln, numerical=True)
        else:
            self.robot.update_plot(pose=pose, soln=soln)
        self.canvas.draw()
        self.canvas.flush_events()


    def activate_VK(self) -> None:
        """
        Activate velocity kinematics mode.

        While VK is active, the robot is continuously updated using the current
        commanded velocity vector `self.v`, which is controlled by the keyboard
        callbacks (`on_press`, `on_release`).

        Returns:
            None. Enters a loop until VK is deactivated.

        Warning:
            This method runs a blocking loop in the UI thread. For more responsive GUIs,
            consider using `root.after(...)` or threading.
        """
        self.vk_status = True
        while self.vk_status:
            self.robot.move_velocity(self.v)
            self.canvas.draw()
            self.canvas.flush_events()
            time.sleep(0.05)


    def deactivate_VK(self) -> None:
        """
        Deactivates velocity kinematics, stopping the robot's movement.
        """
        self.vk_status = False


    def update_waypoints(self) -> None:
        """
        Load waypoints from a YAML file and update the robot visualization.

        This reads `waypoints.yml` from the current working directory and expects
        a structure like:

            points:
              - [x, y, z]
              - [x, y, z]
        """
        print('\nUpdating waypoints...')

        with open('waypoints.yml', 'r') as file:
            waypoints = yaml.safe_load(file)

        self.waypoint_idx = 0
        self.robot.update_waypoints(waypoints['points'])
        self.robot.plot_3D()
        self.canvas.draw()

    
    def generate_traj_task_space(self):
        """
        Generates and visualizes a task-space trajectory using a polynomial interpolator between waypoints.
        """
    
        print('\nFollowing trajectory in task space...')
    
        waypoints = self.robot.get_waypoints()
        q0 = waypoints[0]
        qf = waypoints[1]

        traj = MultiAxisTrajectoryGenerator(  # type: ignore[name-defined]
            method="quintic", mode="task", interval=[0, 1], ndof=len(q0), start_pos=q0, final_pos=qf
        )
        traj_dofs = traj.generate(nsteps=50)

        for i in range(50):
            pos = [dof[0][i] for dof in traj_dofs]
            ee = EndEffector(*pos, 0, -math.pi/2, wraptopi(math.atan2(pos[1], pos[0]) + math.pi))
            self.update_IK(ee, soln=0, numerical=False, display_traj=True)
            time.sleep(0.05)

    
    def generate_traj_joint_space(self) -> None:
        """
        Generates and visualizes a joint-space trajectory by solving inverse kinematics at waypoints
        and interpolating between resulting joint configurations.
        """

        print('\nFollowing trajectory in joint space...')
        
        waypoints = self.robot.get_waypoints()

        EE_0 = EndEffector(*waypoints[0], 0, 0, 0)
        EE_f = EndEffector(*waypoints[1], 0, 0, 0)

        q0 = np.rad2deg(self.robot.solve_inverse_kinematics(EE_0))
        qf = np.rad2deg(self.robot.solve_inverse_kinematics(EE_f))

        traj = MultiAxisTrajectoryGenerator(  # type: ignore[name-defined]
            method="quintic", mode="task", interval=[0, 1], ndof=len(q0), start_pos=q0, final_pos=qf
        )
        traj_dofs = traj.generate(nsteps=50)

        for i in range(50):
            joint_values = [dof[0][i] for dof in traj_dofs]            
            self.update_FK(joint_values=joint_values, display_traj=True) 
            time.sleep(0.05)



    def check_vk_status(self) -> str:
        """
        Checks and returns the status of the velocity kinematics.

        Returns:
            str: The status of velocity kinematics ("Activated!" or "Deactivated!").
        """
        return 'Deactivated!' if not self.vk_status else 'Activated!'


    def on_press(self, key: keyboard.Key) -> None:
        """
        Handles key press events to control the velocity of the robot.

        Args:
            key (pynput.keyboard.Key): The key that was pressed.
        """
        if self.vk_status:
            if key == keyboard.Key.up:
                self.v[1] = 1
            elif key == keyboard.Key.down:
                self.v[1] = -1
            elif key == keyboard.Key.left:
                self.v[0] = -1
            elif key == keyboard.Key.right:
                self.v[0] = 1
            elif hasattr(key, 'char'):
                if key.char == 'w':
                    self.v[2] = 1
                elif key.char == 's':
                    self.v[2] = -1


    def on_release(self, key: keyboard.Key) -> None:
        """
        Handles key release events to stop the robot's movement.

        Args:
            key (pynput.keyboard.Key): The key that was released.
        """
        if key == keyboard.Key.up:
            self.v[1] = 0
        elif key == keyboard.Key.down:
            self.v[1] = 0
        elif key == keyboard.Key.left:
            self.v[0] = 0
        elif key == keyboard.Key.right:
            self.v[0] = 0
        elif hasattr(key, 'char'):
            if key.char == 'w':
                self.v[2] = 0
            elif key.char == 's':
                self.v[2] = 0

    
    def set_pose_values(self, values):
        """
        Populate the IK pose entry fields with specified values.

        Args:
            values: List of 6 values [x, y, z, rotx, roty, rotz]
        """
        if len(values) != 6:
            raise ValueError("Pose must contain exactly 6 values.")

        for entry, val in zip(self.pose_button, values):
            entry.delete(0, tk.END)
            entry.insert(0, str(val))   

    
    def load_current_pose(self):
        """
        Load the current robot pose into the IK entry fields.
        """
        ee = self.robot.model.ee
        pose = [round(val, 4) for val in [ee.x, ee.y, ee.z, ee.rotx, ee.roty, ee.rotz]]

        self.set_pose_values(pose)


    def run(self) -> None:
        """
        Start the Tkinter main loop (if this Visualizer owns the root window).
        """
        self.root.mainloop()



class RobotSim:
    """
    Robot simulation + 3D visualization wrapper for a kinematics model.

    This class wraps a robot model and:
    - Manages a Matplotlib 3D figure
    - Calls the model's kinematics methods to update joint/EE state
    - Draws links, joints, reference frames, and waypoints

    Args:
        robot_model: A robot model providing kinematics methods and state (see module docstring).
        show_animation: Whether to create and update the Matplotlib figure.

    Attributes:
        model: The underlying robot model.
        num_joints: Number of joints/DOF derived from the model.
        fig: Matplotlib Figure (if show_animation=True).
        sub1: 3D axes (if show_animation=True).
        plot_limits: [x_limit, y_limit, z_limit] for display.
        waypoint_x/y/z: Stored waypoint coordinates for visualization.
    """

    def __init__(self, robot_model=None, show_animation: bool=True) -> None:
        """
        Initializes a robot with a specific configuration based on the type.

        Args:
            robot_model: Robot model instance to be visualized and controlled.
            show_animation: If True, creates a Matplotlib figure and renders motion.
        """

        self.model = robot_model
        self.num_joints = robot_model.num_dof        
        
        self.origin = [0., 0., 0.]
        self.axes_length = 0.04
        self.point_x, self.point_y, self.point_z = [], [], []
        self.waypoint_x, self.waypoint_y, self.waypoint_z = [], [], []
        self.waypoint_rotx, self.waypoint_roty, self.waypoint_rotz = [], [], []
        self.show_animation = show_animation
        self.plot_limits = [0.65, 0.65, 0.8]

        if self.show_animation:
            self.fig = Figure(figsize=(12, 10), dpi=100)
            self.sub1 = self.fig.add_subplot(1,1,1, projection='3d') 
            self.fig.suptitle("Manipulator Kinematics Visualization", fontsize=16)

        self.init_plot()

    
    def init_plot(self) -> None:
        """
        Initializes the plot by calculating the robot's points and calling the plot function.
        """
        curr_joint_values = self.get_joint_values()
        _, Hlist = self.model.calc_forward_kinematics(curr_joint_values, radians=True)
        self.model.calc_robot_points(self.model.joint_values, Hlist)
        self.plot_3D()

    
    def update_plot(
        self,
        pose: EndEffector = None,
        joint_values: List[float] = None,
        soln: int = 0,
        numerical: bool = False,
    ) -> None:
        """
        Updates the robot's state based on new pose or joint angles and updates the visualization.

        Args:
            pose: Desired end-effector pose. If provided, IK is used.
            joint_values: Joint values. If provided (and pose is None), FK is used.
            soln: IK solution branch index (for analytical IK). Defaults to 0.
            numerical: If True, use numerical IK when pose is provided.
        """
        if pose is not None: # Inverse kinematics case
            curr_joint_values = self.get_joint_values()
            if not numerical:
                new_joint_values = self.model.calc_inverse_kinematics(pose, curr_joint_values, soln=soln)
                _, Hlist = self.model.calc_forward_kinematics(new_joint_values, radians=True)
            else:
                new_joint_values = self.model.calc_numerical_ik(pose, curr_joint_values, tol=0.02, ilimit=50)
                _, Hlist = self.model.calc_forward_kinematics(new_joint_values, radians=True)
            
            self.model.calc_robot_points(new_joint_values, Hlist)
        
        elif joint_values is not None: # Forward kinematics case
            _, Hlist = self.model.calc_forward_kinematics(joint_values)
            self.model.calc_robot_points(joint_values, Hlist)

        else:
            return
        
        self.plot_3D()


    def move_velocity(self, vel: List[float]) -> None:
        """
        Moves the robot based on a given velocity input.

        Args:
            vel (list): Velocity input for the robot.
        """
        curr_joint_values = self.get_joint_values()
        new_joint_values = self.model.calc_velocity_kinematics(curr_joint_values, vel)
        _, Hlist = self.model.calc_forward_kinematics(new_joint_values)
        self.model.calc_robot_points(new_joint_values, Hlist)
        self.plot_3D()

    
    def reset_ee_trajectory(self):
        self.theta_traj = []
        

    def draw_line_3D(self, p1: List[float], p2: List[float], format_type: str = "k-") -> None:
        """
        Draws a 3D line between two points.

        Args:
            p1 (list): Coordinates of the first point.
            p2 (list): Coordinates of the second point.
            format_type (str, optional): The format of the line. Defaults to "k-".
        """
        self.sub1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], format_type)


    def draw_ref_line(self, point: List[float], axes = None, ref: str = "xyz") -> None:
        """
        Draws reference lines from a given point along specified axes.

        Args:
            point (list): The coordinates of the point to draw from.
            axes (matplotlib.axes, optional): The axes on which to draw the reference lines.
            ref (str, optional): Which reference axes to draw ('xyz', 'xy', or 'xz'). Defaults to 'xyz'.
        """
        line_width = 0.7
        if ref == 'xyz':
            axes.plot([point[0], self.plot_limits[0]],
                      [point[1], point[1]],
                      [point[2], point[2]], 'b--', linewidth=line_width)    # X line
            axes.plot([point[0], point[0]],
                      [point[1], self.plot_limits[1]],
                      [point[2], point[2]], 'b--', linewidth=line_width)    # Y line
            axes.plot([point[0], point[0]],
                      [point[1], point[1]],
                      [point[2], 0.0], 'b--', linewidth=line_width)         # Z line
        elif ref == 'xy':
            axes.plot([point[0], self.plot_limits[0]],
                      [point[1], point[1]], 'b--', linewidth=line_width)    # X line
            axes.plot([point[0], point[0]],
                      [point[1], self.plot_limits[1]], 'b--', linewidth=line_width)    # Y line
        elif ref == 'xz':
            axes.plot([point[0], self.plot_limits[0]],
                      [point[2], point[2]], 'b--', linewidth=line_width)    # X line
            axes.plot([point[0], point[0]],
                      [point[2], 0.0], 'b--', linewidth=line_width)         # Z line


    def plot_waypoints(self) -> None:
        """
        Plots the waypoints in the 3D visualization
        """
        # draw the points
        self.sub1.plot(self.waypoint_x, self.waypoint_y, self.waypoint_z, 'or', markersize=8)


    def update_waypoints(self, waypoints: List[List[float]]) -> None:
        """
        Store waypoint positions for visualization.

        Args:
            waypoints: List of waypoint positions. Each waypoint is expected to contain
                at least [x, y, z].
        """
        for i in range(len(waypoints)):
            self.waypoint_x.append(waypoints[i][0])
            self.waypoint_y.append(waypoints[i][1])
            self.waypoint_z.append(waypoints[i][2])
            # self.waypoint_rotx.append(waypoints[i][3])
            # self.waypoint_roty.append(waypoints[i][4])
            # self.waypoint_rotz.append(waypoints[i][5])


    def plot_3D(self) -> None:
        """
        Redraw the full 3D visualization frame.

        Draws:
            - Robot links and joint points
            - End-effector point and coordinate frame
            - Base coordinate frame
            - Waypoints
            - Reference/trace lines
            - Text overlay with EE pose and joint values
        """        
        self.sub1.cla()
        self.point_x.clear()
        self.point_y.clear()
        self.point_z.clear()

        EE = self.model.ee

        # draw lines to connect the points
        for i in range(len(self.model.points)-1):
            self.draw_line_3D(self.model.points[i], self.model.points[i+1])

        # draw the points
        for i in range(len(self.model.points)):
            self.point_x.append(self.model.points[i][0])
            self.point_y.append(self.model.points[i][1])
            self.point_z.append(self.model.points[i][2])
        self.sub1.plot(self.point_x, self.point_y, self.point_z, marker='o', markerfacecolor='m', markersize=12)

        # draw the waypoints
        self.plot_waypoints()

        # draw the EE
        self.sub1.plot(EE.x, EE.y, EE.z, 'bo')
        # draw the base reference frame
        self.draw_line_3D(self.origin, [self.origin[0] + self.axes_length, self.origin[1], self.origin[2]], format_type='r-')
        self.draw_line_3D(self.origin, [self.origin[0], self.origin[1] + self.axes_length, self.origin[2]], format_type='g-')
        self.draw_line_3D(self.origin, [self.origin[0], self.origin[1], self.origin[2] + self.axes_length], format_type='b-')
        # draw the EE reference frame
        self.draw_line_3D([EE.x, EE.y, EE.z], self.model.EE_axes[0], format_type='r-')
        self.draw_line_3D([EE.x, EE.y, EE.z], self.model.EE_axes[1], format_type='g-')
        self.draw_line_3D([EE.x, EE.y, EE.z], self.model.EE_axes[2], format_type='b-')
        # draw reference / trace lines
        self.draw_ref_line([EE.x, EE.y, EE.z], self.sub1, ref='xyz')

        # add text at bottom of window
        pose_text = "End-effector Pose:      [ "
        pose_text += f"X: {round(EE.x,4)},  "
        pose_text += f"Y: {round(EE.y,4)},  "
        pose_text += f"Z: {round(EE.z,4)},  "
        pose_text += f"RotX: {round(EE.rotx,4)},  "
        pose_text += f"RotY: {round(EE.roty,4)},  "
        pose_text += f"RotZ: {round(EE.rotz,4)}  "
        pose_text += " ]"

        joint_values_text = "Joint Positions (deg/m):     ["
        for i in range(self.num_joints):
            joint_values_text += f" {round(np.rad2deg(self.get_joint_values()[i]),2)}, "
        joint_values_text += " ]"
        
        textstr = pose_text + "\n" + joint_values_text
        self.sub1.text2D(0.2, 0.02, textstr, fontsize=13, transform=self.fig.transFigure)

        self.sub1.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        self.sub1.set_ylim(-self.plot_limits[1], self.plot_limits[1])
        self.sub1.set_zlim(0, self.plot_limits[2])
        self.sub1.set_xlabel('x [m]')
        self.sub1.set_ylabel('y [m]')


    def get_joint_values(self) -> List[float]:
        """
        Return a copy of the model's current joint values.

        Returns:
            list[float]: Joint values (radians internally).
        """
        return self.model.joint_values.copy()