'''
# This module is a simulation environment, which provides different-level (from easy to hard)
# simulation benchmark environments and animation facilities for the user to test their learning algorithm.
# This environment is versatile to use, e.g. the user can arbitrarily:
# set the parameters for the dynamics and objective function,
# obtain the analytical dynamics models, as well as the differentiations.
# define and modify the control cost function
# animate the motion of the system.

# Do NOT use it for any commercial purpose

# Contact email: wanxinjin@gmail.com
# Last update: May. 15, 2020

#

'''

from casadi import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg-master-latest-win64-gpl\bin\\ffmpeg.exe'
import matplotlib.animation as animation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch
import math
import time
import tkinter
import tkinter.messagebox


# multi_UAV formation environment
class UAV_formation_obst:
    def __init__(self, project_name='my UAV swarm'):
        self.project_name = 'my uav swarm'

        self.states_bd = [None, None]
        self.ctrl_bd = [None, None]
        self.obstacles = [[1.5, 9], [3.75, 5], [5, 5]]
        self.wall = [0, 5]

    def initDyn(self, uav_index=None, Jx=None, Jy=None, Jz=None, mass=None, l=None, c=None, states_bd=[None, None],
                ctrl_bd=[None, None], dt=0.1):
        # define the state of the uavs
        rx, ry, rz = SX.sym('rx' + str(uav_index)), SX.sym('ry' + str(uav_index)), SX.sym('rz' + str(uav_index))
        self.r_I = vertcat(rx, ry, rz)
        vx, vy, vz = SX.sym('vx' + str(uav_index)), SX.sym('vy' + str(uav_index)), SX.sym('vz' + str(uav_index))
        self.v_I = vertcat(vx, vy, vz)
        # quaternions attitude of B w.r.t. I
        q0, q1, q2, q3 = SX.sym('q0' + str(uav_index)), SX.sym('q1' + str(uav_index)), SX.sym(
            'q2' + str(uav_index)), SX.sym('q3' + str(uav_index))
        self.q = vertcat(q0, q1, q2, q3)
        wx, wy, wz = SX.sym('wx' + str(uav_index)), SX.sym('wy' + str(uav_index)), SX.sym('wz' + str(uav_index))
        self.w_B = vertcat(wx, wy, wz)
        # define the quadrotor input
        f1, f2, f3, f4 = SX.sym('f1' + str(uav_index)), SX.sym('f2' + str(uav_index)), SX.sym(
            'f3' + str(uav_index)), SX.sym('f4' + str(uav_index))
        self.T_B = vertcat(f1, f2, f3, f4)

        # global parameter
        g = 10

        # parameters settings
        parameter = []
        if Jx is None:
            self.Jx = SX.sym('Jx' + str(uav_index))
            parameter += [self.Jx]
        else:
            self.Jx = Jx

        if Jy is None:
            self.Jy = SX.sym('Jy' + str(uav_index))
            parameter += [self.Jy]
        else:
            self.Jy = Jy

        if Jz is None:
            self.Jz = SX.sym('Jz' + str(uav_index))
            parameter += [self.Jz]
        else:
            self.Jz = Jz

        if mass is None:
            self.mass = SX.sym('mass' + str(uav_index))
            parameter += [self.mass]
        else:
            self.mass = mass

        if l is None:
            self.l = SX.sym('l' + str(uav_index))
            parameter += [self.l]
        else:
            self.l = l

        if c is None:
            self.c = SX.sym('c' + str(uav_index))
            parameter += [self.c]
        else:
            self.c = c

        self.dyn_auxvar = vcat(parameter)

        # Angular moment of inertia
        self.J_B = diag(vertcat(self.Jx, self.Jy, self.Jz))
        # Gravity
        self.g_I = vertcat(0, 0, -g)
        # Mass of rocket, assume is little changed during the landing process
        self.m = self.mass

        # total thrust in body frame
        thrust = self.T_B[0] + self.T_B[1] + self.T_B[2] + self.T_B[3]
        self.thrust_B = vertcat(0, 0, thrust)
        # total moment M in body frame
        Mx = -self.T_B[1] * self.l / 2 + self.T_B[3] * self.l / 2
        My = -self.T_B[0] * self.l / 2 + self.T_B[2] * self.l / 2
        Mz = (self.T_B[0] - self.T_B[1] + self.T_B[2] - self.T_B[3]) * self.c
        self.M_B = vertcat(Mx, My, Mz)

        # cosine directional matrix
        C_B_I = self.dir_cosine(self.q)  # inertial to body
        C_I_B = transpose(C_B_I)  # body to inertial

        # Newton's law
        dr_I = self.v_I
        dv_I = 1 / self.m * mtimes(C_I_B, self.thrust_B) + self.g_I
        # Euler's law
        dq = 1 / 2 * mtimes(self.omega(self.w_B), self.q)
        dw = mtimes(inv(self.J_B), self.M_B - mtimes(mtimes(self.skew(self.w_B), self.J_B), self.w_B))

        self.states = vertcat(self.r_I, self.v_I, self.q, self.w_B)

        if np.asarray(states_bd).size == self.states.numel():
            self.states_bd = states_bd
        else:
            self.states_bd = self.states.numel() * [-1e10, 1e10]

        self.ctrl = self.T_B
        if np.asarray(ctrl_bd).size == self.ctrl.numel():
            self.ctrl_bd = ctrl_bd
        else:
            self.states_bd = self.states.numel() * [-1e10, 1e10]

        self.dynf = vertcat(dr_I, dv_I, dq, dw)
        self.dt = dt

    def initCost(self, uavswarm=None, uav_index=None, w_F_r=None, w_F_v=None, w_r_formation=None, w_v_formation=None,
                 w_uva_collision=None, w_obst=None, w_altitude=10, wthrust=0.1):

        self.w_altitude = w_altitude
        self.goal_r_I = np.array([uav_index + 1, 15, 5])
        self.goal_v_I = np.array([0, 3, 0])

        parameter = []
        if w_F_r is None:
            self.w_F_r = SX.sym('w_F_r' + str(uav_index))
            parameter += [self.w_F_r]
        else:
            self.w_F_r = w_F_r

        if w_F_v is None:
            self.w_F_v = SX.sym('w_F_v' + str(uav_index))
            parameter += [self.w_F_v]
        else:
            self.w_F_v = w_F_v

        if w_r_formation is None:
            self.w_r_formation = SX.sym('w_r_formation' + str(uav_index))
            parameter += [self.w_r_formation]
        else:
            self.w_r_formation = w_r_formation

        if w_v_formation is None:
            self.w_v_formation = SX.sym('w_v_formation' + str(uav_index))
            parameter += [self.w_v_formation]
        else:
            self.w_v_formation = w_v_formation

        if w_uva_collision is None:
            self.w_uva_collision = SX.sym('w_uva_collision' + str(uav_index))
            parameter += [self.w_uva_collision]
        else:
            self.w_uva_collision = w_uva_collision

        if w_obst is None:
            self.w_obst = SX.sym('w_obst' + str(uav_index))
            parameter += [self.w_obst]
        else:
            self.w_obst = w_obst

        self.cost_auxvar = vcat(parameter)

        # neighbors
        dist = [1, 0, 0]
        neighbor_p = uav_index + 1 if (uav_index + 1) < len(uavswarm) else uav_index + 1 - len(uavswarm)
        neighbor_m = uav_index - 1 if (uav_index - 1) >= 0 else uav_index - 1 + len(uavswarm)
        # formation distance and cost
        self.cost_r_formation = norm_2(self.r_I - uavswarm[neighbor_p].r_I + [neighbor_p - uav_index, 0, 0]) + \
                                norm_2(self.r_I - uavswarm[neighbor_m].r_I + [neighbor_m - uav_index, 0, 0])

        # formation speed cost
        self.cost_v_formation = norm_2(self.v_I - uavswarm[neighbor_p].v_I) + \
                                norm_2(self.v_I - uavswarm[neighbor_m].v_I)

        goal_height = 5
        self.cost_altitude = dot(self.r_I[2] - goal_height, self.r_I[2] - goal_height)

        # formation attitude cost
        '''
        goal_q = toQuaternion(0, [0, 0, 1])
        goal_R_B_I = self.dir_cosine(goal_q)
        R_B_I = self.dir_cosine(self.q)
        self.cost_q = trace(np.identity(3) - mtimes(transpose(goal_R_B_I), R_B_I))
        '''

        # goal position cost in the world frame
        # if uav_index == 0 or uav_index == 3:
        self.cost_r_I = norm_2(self.r_I - self.goal_r_I)
        # else:
        #    self.cost_r_I = 0
        # goal velocity cost
        if uav_index == 0:
            self.cost_v_I = norm_2(self.v_I - self.goal_v_I)
        else:
            self.cost_v_I = 0
        # collision cost

        # self.cost_collision = 0
        # for i in range(len(uavswarm)):
        #     if i != uav_index:
        #         self.cost_collision = self.cost_collision + 1 / dot(self.r_I - uavswarm[i].r_I,
        #                                                             self.r_I - uavswarm[i].r_I)
        self.cost_collision = 1 / fmin(fmax(norm_2(self.r_I - uavswarm[neighbor_p].r_I), 1e-3), 0.8) - 1 / 0.8 + \
                              1 / fmin(fmax(norm_2(self.r_I - uavswarm[neighbor_m].r_I), 1e-3), 0.8) - 1 / 0.8

        # self.cost_obst = 5 * fmax(self.wall[0] - self.r_I[0], 0) + 5 * fmax(self.r_I[0] - self.wall[1], 0) + \
        #                 1 / fmax(
        #    fmin(norm_1(self.r_I[0:2] - self.obstacles[0], self.r_I[0:2] - self.obstacles[0]), 1e-3), 0.25) + \
        #                 1 / fmax(
        #    fmin(norm_1(self.r_I[0:2] - self.obstacles[1], self.r_I[0:2] - self.obstacles[1]), 1e-3), 0.25)
        self.cost_obst = 5 * fmax(self.wall[0] - self.r_I[0], 0) + 5 * fmax(self.r_I[0] - self.wall[1], 0) + \
                         1 / fmin(fmax(norm_2(self.r_I[0:2] - self.obstacles[0]), 1e-3), 0.4) - 1 / 0.4 + \
                         1 / (fmin(fmax(norm_2(self.r_I[0:2] - self.obstacles[1]), 1e-3), 1)) - 1 + \
                         1 / (fmin(fmax(norm_2(self.r_I[0:2] - self.obstacles[2]), 1e-3), 1)) - 1

        # auglar velocity cost
        '''
        goal_w_B = np.array([0, 0, 0])
        self.cost_w_B = dot(self.w_B - goal_w_B, self.w_B - goal_w_B)
        '''

        # the thrust cost
        self.cost_thrust = dot(self.T_B, self.T_B)

        self.path_cost = self.w_r_formation * self.cost_r_formation + \
                         self.w_v_formation * self.cost_v_formation + \
                         self.w_uva_collision * self.cost_collision + \
                         self.w_altitude * self.cost_altitude + \
                         self.w_obst * self.cost_obst + \
                         wthrust * self.cost_thrust
        # self.w_F_r * self.cost_r_I + \
        # self.w_F_v * self.cost_v_I + \

        self.final_cost = self.w_F_r * self.cost_r_I + \
                          self.w_F_v * self.cost_v_I + \
                          10 * self.w_r_formation * self.cost_r_formation + \
                          3 * self.w_v_formation * self.cost_v_formation + \
                          self.w_altitude * self.cost_altitude + \
                          self.w_uva_collision * self.cost_collision

    def get_UAV_position(self, wing_len, state_traj):

        # thrust_position in body frame
        r1 = vertcat(wing_len / 2, 0, 0)
        r2 = vertcat(0, -wing_len / 2, 0)
        r3 = vertcat(-wing_len / 2, 0, 0)
        r4 = vertcat(0, wing_len / 2, 0)

        # horizon
        horizon = np.size(state_traj, 0)
        position = np.zeros((horizon, 15))
        for t in range(horizon):
            # position of COM
            rc = state_traj[t, 0:3]
            # altitude of quaternion
            q = state_traj[t, 6:10]

            # direction cosine matrix from body to inertial
            CIB = np.transpose(self.dir_cosine(q).full())

            # position of each rotor in inertial frame
            r1_pos = rc + mtimes(CIB, r1).full().flatten()
            r2_pos = rc + mtimes(CIB, r2).full().flatten()
            r3_pos = rc + mtimes(CIB, r3).full().flatten()
            r4_pos = rc + mtimes(CIB, r4).full().flatten()

            # store
            position[t, 0:3] = rc
            position[t, 3:6] = r1_pos
            position[t, 6:9] = r2_pos
            position[t, 9:12] = r3_pos
            position[t, 12:15] = r4_pos

        return position

    def play_multi_animation(self, uavswarm, wing_len, state_traj, state_traj_ref=None, dt=0.1, save_option=0,
                             title='UAV Maneuvering'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_zlabel('Z (m)', fontsize=10, labelpad=5)
        ax.set_zlim(0, 10)
        ax.set_ylim(-1, 16)
        ax.set_xlim(-1, 7)
        ax.set_box_aspect(aspect=(8, 17, 10))
        ax.set_title(title, pad=20, fontsize=15)

        # target landing point
        # ax.scatter3D([0.0], [0.0], [0.0], c="r", marker="x")
        # environment
        def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
            z = np.linspace(0, height_z, 50)
            theta = np.linspace(0, 2 * np.pi, 50)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = radius * np.cos(theta_grid) + center_x
            y_grid = radius * np.sin(theta_grid) + center_y
            return x_grid, y_grid, z_grid

        def data_for_cube(o, size=(1, 1, 1)):
            l, w, h = size
            x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],
                 [o[0], o[0] + l, o[0] + l, o[0], o[0]],
                 [o[0], o[0] + l, o[0] + l, o[0], o[0]],
                 [o[0], o[0] + l, o[0] + l, o[0], o[0]]]
            y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
                 [o[1], o[1], o[1] + w, o[1] + w, o[1]],
                 [o[1], o[1], o[1], o[1], o[1]],
                 [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
            z = [[o[2], o[2], o[2], o[2], o[2]],
                 [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
                 [o[2], o[2], o[2] + h, o[2] + h, o[2]],
                 [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
            return np.array(x), np.array(y), np.array(z)

        OXc, OYc, OZc = data_for_cylinder_along_z(self.obstacles[0][0], self.obstacles[0][1], 0.2, 10)
        ax.plot_surface(OXc, OYc, OZc, alpha=0.5)
        OXc, OYc, OZc = data_for_cube([self.obstacles[1][0] - 0.75, self.obstacles[1][1] - 0.5, 0], size=[2, 1, 10])
        ax.plot_surface(OXc, OYc, OZc, alpha=0.5)

        OXc, OYc, OZc = data_for_cube([0, -1, 0], size=[0.1, 11, 10])
        ax.plot_surface(OXc, OYc, OZc, alpha=0.1, color='limegreen')

        OXc, OYc, OZc = data_for_cube([5, -1, 0], size=[0.1, 11, 10])
        ax.plot_surface(OXc, OYc, OZc, alpha=0.1, color='limegreen')

        # data
        self.num_uav = len(uavswarm)
        position = [None] * self.num_uav
        position_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            position[i] = self.get_UAV_position(wing_len, state_traj[i])
            if state_traj_ref is None:
                position_ref[i] = self.get_UAV_position(0, numpy.zeros_like(position[i]))
            else:
                position_ref[i] = self.get_UAV_position(wing_len, state_traj_ref[i])

        # animation
        line_traj = [None] * self.num_uav
        c_x = [None] * self.num_uav
        c_y = [None] * self.num_uav
        c_z = [None] * self.num_uav
        r1_x = [None] * self.num_uav
        r1_y = [None] * self.num_uav
        r1_z = [None] * self.num_uav
        r2_x = [None] * self.num_uav
        r2_y = [None] * self.num_uav
        r2_z = [None] * self.num_uav
        r3_x = [None] * self.num_uav
        r3_y = [None] * self.num_uav
        r3_z = [None] * self.num_uav
        r4_x = [None] * self.num_uav
        r4_y = [None] * self.num_uav
        r4_z = [None] * self.num_uav
        line_arm1 = [None] * self.num_uav
        line_arm2 = [None] * self.num_uav
        line_arm3 = [None] * self.num_uav
        line_arm4 = [None] * self.num_uav
        line_traj_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            line_traj[i], = ax.plot(position[i][:1, 0], position[i][:1, 1], position[i][:1, 2])
            c_x[i], c_y[i], c_z[i] = position[i][0, 0:3]
            r1_x[i], r1_y[i], r1_z[i] = position[i][0, 3:6]
            r2_x[i], r2_y[i], r2_z[i] = position[i][0, 6:9]
            r3_x[i], r3_y[i], r3_z[i] = position[i][0, 9:12]
            r4_x[i], r4_y[i], r4_z[i] = position[i][0, 12:15]
            line_arm1[i], = ax.plot([c_x[i], r1_x[i]], [c_y[i], r1_y[i]], [c_z[i], r1_z[i]], linewidth=1, color='red',
                                    marker='o', markersize=2)
            line_arm2[i], = ax.plot([c_x[i], r2_x[i]], [c_y[i], r2_y[i]], [c_z[i], r2_z[i]], linewidth=1, color='blue',
                                    marker='o', markersize=2)
            line_arm3[i], = ax.plot([c_x[i], r3_x[i]], [c_y[i], r3_y[i]], [c_z[i], r3_z[i]], linewidth=1, color='red',
                                    marker='o', markersize=2)
            line_arm4[i], = ax.plot([c_x[i], r4_x[i]], [c_y[i], r4_y[i]], [c_z[i], r4_z[i]], linewidth=1, color='blue',
                                    marker='o', markersize=2)

            # line_traj_ref[i], = ax.plot(position_ref[:1, 0], position_ref[:1, 1], position_ref[:1, 2], color='gray', alpha=0.5)
            # c_x_ref, c_y_ref, c_z_ref = position_ref[0, 0:3]
            # r1_x_ref, r1_y_ref, r1_z_ref = position_ref[0, 3:6]
            # r2_x_ref, r2_y_ref, r2_z_ref = position_ref[0, 6:9]
            # r3_x_ref, r3_y_ref, r3_z_ref = position_ref[0, 9:12]
            # r4_x_ref, r4_y_ref, r4_z_ref = position_ref[0, 12:15]
            # line_arm1_ref, = ax.plot([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref], [c_z_ref, r1_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm2_ref, = ax.plot([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref], [c_z_ref, r2_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm3_ref, = ax.plot([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref], [c_z_ref, r3_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm4_ref, = ax.plot([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref], [c_z_ref, r4_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)

        # time label
        time_template = 'time = %.1fs'
        time_text = ax.text2D(0.66, 0.55, "time", transform=ax.transAxes)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))
        sim_horizon = np.size(position[0], 0)

        def update_traj(num):
            # customize
            time_text.set_text(time_template % (num * dt))

            # trajectory
            for i in range(self.num_uav):
                line_traj[i].set_data(position[i][:num, 0], position[i][:num, 1])
                line_traj[i].set_3d_properties(position[i][:num, 2])

                # uav
                c_x[i], c_y[i], c_z[i] = position[i][num, 0:3]
                r1_x[i], r1_y[i], r1_z[i] = position[i][num, 3:6]
                r2_x[i], r2_y[i], r2_z[i] = position[i][num, 6:9]
                r3_x[i], r3_y[i], r3_z[i] = position[i][num, 9:12]
                r4_x[i], r4_y[i], r4_z[i] = position[i][num, 12:15]

                line_arm1[i].set_data(np.array([[c_x[i], r1_x[i]], [c_y[i], r1_y[i]]]))
                line_arm1[i].set_3d_properties([c_z[i], r1_z[i]])

                line_arm2[i].set_data(np.array([[c_x[i], r2_x[i]], [c_y[i], r2_y[i]]]))
                line_arm2[i].set_3d_properties([c_z[i], r2_z[i]])

                line_arm3[i].set_data(np.array([[c_x[i], r3_x[i]], [c_y[i], r3_y[i]]]))
                line_arm3[i].set_3d_properties([c_z[i], r3_z[i]])

                line_arm4[i].set_data(np.array([[c_x[i], r4_x[i]], [c_y[i], r4_y[i]]]))
                line_arm4[i].set_3d_properties([c_z[i], r4_z[i]])

            # trajectory ref
            #     num = sim_horizon - 1
            #     line_traj_ref.set_data(position_ref[:num, 0], position_ref[:num, 1])
            #     line_traj_ref.set_3d_properties(position_ref[:num, 2])
            #
            #     # uav ref
            #     c_x_ref, c_y_ref, c_z_ref = position_ref[num, 0:3]
            #     r1_x_ref, r1_y_ref, r1_z_ref = position_ref[num, 3:6]
            #     r2_x_ref, r2_y_ref, r2_z_ref = position_ref[num, 6:9]
            #     r3_x_ref, r3_y_ref, r3_z_ref = position_ref[num, 9:12]
            #     r4_x_ref, r4_y_ref, r4_z_ref = position_ref[num, 12:15]
            #
            #     line_arm1_ref.set_data(np.array([[c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref]]))
            #     line_arm1_ref.set_3d_properties([c_z_ref, r1_z_ref])
            #
            #     line_arm2_ref.set_data(np.array([[c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref]]))
            #     line_arm2_ref.set_3d_properties([c_z_ref, r2_z_ref])
            #
            #     line_arm3_ref.set_data(np.array([[c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref]]))
            #     line_arm3_ref.set_3d_properties([c_z_ref, r3_z_ref])
            #
            #     line_arm4_ref.set_data(np.array([[c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref]]))
            #     line_arm4_ref.set_3d_properties([c_z_ref, r4_z_ref])

            # if num == sim_horizon-2:
            #    plt.pause(2)

            return line_traj[0], line_arm1[0], line_arm2[0], line_arm3[0], line_arm4[0], line_traj[1], line_arm1[1], \
                   line_arm2[1], line_arm3[1], line_arm4[1], line_traj[2], line_arm1[2], line_arm2[2], line_arm3[2], \
                   line_arm4[2], line_traj[3], line_arm1[3], line_arm2[3], line_arm3[3], line_arm4[3], time_text  # , \

            # line_traj_ref, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=100, blit=True)

        def restart():
            root = tkinter.Tk()
            root.withdraw()
            result = tkinter.messagebox.askyesno("Restart", "Do you want to restart animation?")
            if result:
                ani.frame_seq = ani.new_frame_seq()
                ani.event_source.start()
            else:
                plt.close()

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('case2' + title + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()

    def dir_cosine(self, q):
        C_B_I = vertcat(
            horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
        return C_B_I

    def skew(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2], v[1]),
            horzcat(v[2], 0, -v[0]),
            horzcat(-v[1], v[0], 0)
        )
        return v_cross

    def omega(self, w):
        omeg = vertcat(
            horzcat(0, -w[0], -w[1], -w[2]),
            horzcat(w[0], 0, w[2], -w[1]),
            horzcat(w[1], -w[2], 0, w[0]),
            horzcat(w[2], w[1], -w[0], 0)
        )
        return omeg

    def quaternion_mul(self, p, q):
        return vertcat(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                       p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                       p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                       p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
                       )


class UAV_formation:
    def __init__(self, project_name='my UAV swarm'):
        self.project_name = 'my uav swarm'

        self.states_bd = [None, None]
        self.ctrl_bd = [None, None]
        self.UAV_labels = None
        self.obstacles = [[4.3, 3], [4.8, 3]]
        self.wall = [0, 5]

    def initDyn(self, uav_index=0, Jx=None, Jy=None, Jz=None, mass=None, l=0.1, c=None, states_bd=[None, None],
                ctrl_bd=[None, None], dt=1e-3):
        # define the state of the uavs
        rx, ry, rz = SX.sym('rx' + str(uav_index)), SX.sym('ry' + str(uav_index)), SX.sym('rz' + str(uav_index))
        self.r_I = vertcat(rx, ry, rz)
        vx, vy, vz = SX.sym('vx' + str(uav_index)), SX.sym('vy' + str(uav_index)), SX.sym('vz' + str(uav_index))
        self.v_I = vertcat(vx, vy, vz)
        # quaternions attitude of B w.r.t. I
        q0, q1, q2, q3 = SX.sym('q0' + str(uav_index)), SX.sym('q1' + str(uav_index)), SX.sym(
            'q2' + str(uav_index)), SX.sym('q3' + str(uav_index))
        self.q = vertcat(q0, q1, q2, q3)
        wx, wy, wz = SX.sym('wx' + str(uav_index)), SX.sym('wy' + str(uav_index)), SX.sym('wz' + str(uav_index))
        self.w_B = vertcat(wx, wy, wz)
        # define the quadrotor input
        f1, f2, f3, f4 = SX.sym('f1' + str(uav_index)), SX.sym('f2' + str(uav_index)), SX.sym(
            'f3' + str(uav_index)), SX.sym('f4' + str(uav_index))
        self.T_B = vertcat(f1, f2, f3, f4)

        # global parameter
        g = 10

        # parameters settings
        parameter = []
        if Jx is None:
            self.Jx = SX.sym('Jx' + str(uav_index))
            parameter += [self.Jx]
        else:
            self.Jx = Jx

        if Jy is None:
            self.Jy = SX.sym('Jy' + str(uav_index))
            parameter += [self.Jy]
        else:
            self.Jy = Jy

        if Jz is None:
            self.Jz = SX.sym('Jz' + str(uav_index))
            parameter += [self.Jz]
        else:
            self.Jz = Jz

        if mass is None:
            self.mass = SX.sym('mass' + str(uav_index))
            parameter += [self.mass]
        else:
            self.mass = mass

        if l is None:
            self.l = SX.sym('l' + str(uav_index))
            parameter += [self.l]
        else:
            self.l = l

        if c is None:
            self.c = SX.sym('c' + str(uav_index))
            parameter += [self.c]
        else:
            self.c = c

        self.dyn_auxvar = vcat(parameter)

        # Angular moment of inertia
        self.J_B = diag(vertcat(self.Jx, self.Jy, self.Jz))
        # Gravity
        self.g_I = vertcat(0, 0, -g)
        # Mass of rocket, assume is little changed during the landing process
        self.m = self.mass

        # total thrust in body frame
        thrust = self.T_B[0] + self.T_B[1] + self.T_B[2] + self.T_B[3]
        self.thrust_B = vertcat(0, 0, thrust)
        # total moment M in body frame
        Mx = -self.T_B[1] * self.l / 2 + self.T_B[3] * self.l / 2
        My = -self.T_B[0] * self.l / 2 + self.T_B[2] * self.l / 2
        Mz = (self.T_B[0] - self.T_B[1] + self.T_B[2] - self.T_B[3]) * self.c
        self.M_B = vertcat(Mx, My, Mz)

        # cosine directional matrix
        C_B_I = self.dir_cosine(self.q)  # inertial to body
        C_I_B = transpose(C_B_I)  # body to inertial

        # Newton's law
        dr_I = self.v_I
        dv_I = 1 / self.m * mtimes(C_I_B, self.thrust_B) + self.g_I
        # Euler's law
        dq = 1 / 2 * mtimes(self.omega(self.w_B), self.q)
        dw = mtimes(inv(self.J_B), self.M_B - mtimes(mtimes(self.skew(self.w_B), self.J_B), self.w_B))

        self.states = vertcat(self.r_I, self.v_I, self.q, self.w_B)

        if np.asarray(states_bd).size == self.states.numel():
            self.states_bd = states_bd
        else:
            self.states_bd = self.states.numel() * [-1e10, 1e10]

        self.ctrl = self.T_B
        if np.asarray(ctrl_bd).size == self.ctrl.numel():
            self.ctrl_bd = ctrl_bd
        else:
            self.states_bd = self.states.numel() * [-1e10, 1e10]

        self.dynf = vertcat(dr_I, dv_I, dq, dw)
        self.dt = dt

    def initCost(self, uavswarm=None, num_uav=None, uav_index=None, w_F_r=None, w_F_v=None, w_r_formation=None,
                 w_v_formation=None,
                 uav_dist=None, w_uav_collision=None, w_obst=None, w_altitude=50, wthrust=0.1):

        parameter = []
        if w_F_r is None:
            self.w_F_r = SX.sym('w_F_r' + str(uav_index))
            parameter += [self.w_F_r]
        else:
            self.w_F_r = w_F_r

        if w_F_v is None:
            self.w_F_v = SX.sym('w_F_v' + str(uav_index))
            parameter += [self.w_F_v]
        else:
            self.w_F_v = w_F_v

        if w_r_formation is None:
            self.w_r_formation = SX.sym('w_r_formation' + str(uav_index))
            parameter += [self.w_r_formation]
        else:
            self.w_r_formation = w_r_formation

        if w_v_formation is None:
            self.w_v_formation = SX.sym('w_v_formation' + str(uav_index))
            parameter += [self.w_v_formation]
        else:
            self.w_v_formation = w_v_formation

        if uav_dist is None:
            self.uav_dist = SX.sym('uav_dist' + str(uav_index))
            parameter += [self.uav_dist]
        else:
            self.uav_dist = uav_dist

        if w_uav_collision is None:
            self.w_uav_collision = SX.sym('w_uav_collision' + str(uav_index))
            parameter += [self.w_uav_collision]
        else:
            self.w_uav_collision = w_uav_collision

        if w_obst is None:
            self.w_obst = SX.sym('w_obst' + str(uav_index))
            parameter += [self.w_obst]
        else:
            self.w_obst = w_obst

        self.cost_auxvar = vcat(parameter)

        # neighbors

        # formation distance and speed costs
        neighbors = [*range(num_uav)]
        neighbors.remove(uav_index)
        self.cost_r_formation = 0
        self.cost_v_formation = 0
        if uav_index == 1:
            self.cost_r_formation = norm_2(self.r_I - uavswarm[3].r_I - vcat([(-2) * self.uav_dist, 0, 0]))
            self.cost_v_formation = dot(self.v_I - uavswarm[3].v_I, self.v_I - uavswarm[3].v_I)
        if uav_index == 3:
            self.cost_r_formation = norm_2(self.r_I - uavswarm[1].r_I - vcat([(2) * self.uav_dist, 0, 0]))
            self.cost_v_formation = dot(self.v_I - uavswarm[1].v_I, self.v_I - uavswarm[1].v_I)
        # for i in neighbors:
        #    self.cost_v_formation = self.cost_v_formation + norm_2(self.v_I - uavswarm[i].v_I)

        goal_height = 5
        self.cost_altitude = dot(self.r_I[2] - goal_height, self.r_I[2] - goal_height)

        # if uav_index == 0:
        goal_r_I = np.array([1, 7, 5])
        self.cost_r_I = norm_2(self.r_I - goal_r_I - vcat([uav_index * self.uav_dist, 0, 0]))

        # else:
        #    self.cost_r_I =0

        #    self.cost_r_I = 0
        # goal velocity cost
        goal_v_I = np.array([0, 2, 0])
        self.cost_v_I = norm_2(self.v_I - goal_v_I)

        # collision cost
        self.cost_collision = 0
        for i in neighbors:
            self.cost_collision = self.cost_collision + 1 / fmin(fmax(norm_2(self.r_I - uavswarm[i].r_I), 1e-3),
                                                                 0.6) - 1 / 0.6

        self.cost_obst = 5 * fmax(self.wall[0] - self.r_I[0], 0) + 5 * fmax(self.r_I[0] - self.wall[1], 0) + \
                         1 / (fmin(fmax(norm_2(self.r_I[0:2] - self.obstacles[0]), 1e-3), 0.6)) - 1 / 0.6 + \
                         1 / (fmin(fmax(norm_2(self.r_I[0:2] - self.obstacles[1]), 1e-3), 0.6)) - 1 / 0.6

        # the thrust cost
        self.cost_thrust = dot(self.T_B, self.T_B)

        self.w_altitude = w_altitude

        self.path_cost = self.w_r_formation * self.cost_r_formation + \
                         self.w_v_formation * self.cost_v_formation + \
                         self.w_uav_collision * self.cost_collision + \
                         self.w_altitude * self.cost_altitude + \
                         10 * self.w_obst * self.cost_obst + \
                         wthrust * self.cost_thrust
        # self.w_v_formation * self.cost_v_formation + \
        # self.w_F_r * self.cost_r_I + \
        # self.w_F_v * self.cost_v_I + \

        self.final_cost = self.w_F_r * self.cost_r_I + \
                          self.w_F_v * self.cost_v_I + \
                          100 * self.w_r_formation * self.cost_r_formation + \
                          100 * self.w_v_formation * self.cost_v_formation + \
                          self.w_altitude * self.cost_altitude + \
                          self.w_uav_collision * self.cost_collision

    def get_UAV_position(self, wing_len, state_traj):

        # thrust_position in body frame
        r1 = vertcat(wing_len / 2, 0, 0)
        r2 = vertcat(0, -wing_len / 2, 0)
        r3 = vertcat(-wing_len / 2, 0, 0)
        r4 = vertcat(0, wing_len / 2, 0)

        # horizon
        horizon = np.size(state_traj, 0)
        position = np.zeros((horizon, 15))
        for t in range(horizon):
            # position of COM
            rc = state_traj[t, 0:3]
            # altitude of quaternion
            q = state_traj[t, 6:10]

            # direction cosine matrix from body to inertial
            CIB = np.transpose(self.dir_cosine(q).full())

            # position of each rotor in inertial frame
            r1_pos = rc + mtimes(CIB, r1).full().flatten()
            r2_pos = rc + mtimes(CIB, r2).full().flatten()
            r3_pos = rc + mtimes(CIB, r3).full().flatten()
            r4_pos = rc + mtimes(CIB, r4).full().flatten()

            # store
            position[t, 0:3] = rc
            position[t, 3:6] = r1_pos
            position[t, 6:9] = r2_pos
            position[t, 9:12] = r3_pos
            position[t, 12:15] = r4_pos

        return position

    def play_multi_animation(self, uavswarm, wing_len, state_traj, state_traj_ref=None, dt=None, save_option=0,
                             title='UAV Maneuvering'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_zlabel('Z (m)', fontsize=10, labelpad=5)
        ax.set_zlim(0, 10)
        ax.set_ylim(-1, 10)
        ax.set_xlim(-1, 6)
        ax.set_box_aspect(aspect=(7, 11, 10))
        ax.set_title(title, pad=20, fontsize=15)

        # target landing point
        # ax.scatter3D([0.0], [0.0], [0.0], c="r", marker="x")
        # environment
        def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
            z = np.linspace(0, height_z, 50)
            theta = np.linspace(0, 2 * np.pi, 50)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = radius * np.cos(theta_grid) + center_x
            y_grid = radius * np.sin(theta_grid) + center_y
            return x_grid, y_grid, z_grid

        def data_for_cube(o, size=(1, 1, 1)):
            l, w, h = size
            x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],
                 [o[0], o[0] + l, o[0] + l, o[0], o[0]],
                 [o[0], o[0] + l, o[0] + l, o[0], o[0]],
                 [o[0], o[0] + l, o[0] + l, o[0], o[0]]]
            y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
                 [o[1], o[1], o[1] + w, o[1] + w, o[1]],
                 [o[1], o[1], o[1], o[1], o[1]],
                 [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
            z = [[o[2], o[2], o[2], o[2], o[2]],
                 [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
                 [o[2], o[2], o[2] + h, o[2] + h, o[2]],
                 [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
            return np.array(x), np.array(y), np.array(z)

        OXc, OYc, OZc = data_for_cube([self.obstacles[0][0] - 0.5, self.obstacles[0][1] - 0.8, 0], size=[1.5, 1, 10])
        ax.plot_surface(OXc, OYc, OZc, alpha=0.5)

        OXc, OYc, OZc = data_for_cube([0, -1, 0], size=[0.1, 11, 10])
        ax.plot_surface(OXc, OYc, OZc, alpha=0.2, color='limegreen')

        OXc, OYc, OZc = data_for_cube([5, -1, 0], size=[0.1, 11, 10])
        ax.plot_surface(OXc, OYc, OZc, alpha=0.2, color='limegreen')

        # data
        self.num_uav = len(uavswarm)
        position = [None] * self.num_uav
        position_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            position[i] = self.get_UAV_position(wing_len, state_traj[i])
            if state_traj_ref is None:
                position_ref[i] = self.get_UAV_position(0, numpy.zeros_like(position[i]))
            else:
                position_ref[i] = self.get_UAV_position(wing_len, state_traj_ref[i])

        # animation
        line_traj = [None] * self.num_uav
        c_x = [None] * self.num_uav
        c_y = [None] * self.num_uav
        c_z = [None] * self.num_uav
        r1_x = [None] * self.num_uav
        r1_y = [None] * self.num_uav
        r1_z = [None] * self.num_uav
        r2_x = [None] * self.num_uav
        r2_y = [None] * self.num_uav
        r2_z = [None] * self.num_uav
        r3_x = [None] * self.num_uav
        r3_y = [None] * self.num_uav
        r3_z = [None] * self.num_uav
        r4_x = [None] * self.num_uav
        r4_y = [None] * self.num_uav
        r4_z = [None] * self.num_uav
        line_arm1 = [None] * self.num_uav
        line_arm2 = [None] * self.num_uav
        line_arm3 = [None] * self.num_uav
        line_arm4 = [None] * self.num_uav
        line_traj_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            line_traj[i], = ax.plot(position[i][:1, 0], position[i][:1, 1], position[i][:1, 2])
            c_x[i], c_y[i], c_z[i] = position[i][0, 0:3]
            r1_x[i], r1_y[i], r1_z[i] = position[i][0, 3:6]
            r2_x[i], r2_y[i], r2_z[i] = position[i][0, 6:9]
            r3_x[i], r3_y[i], r3_z[i] = position[i][0, 9:12]
            r4_x[i], r4_y[i], r4_z[i] = position[i][0, 12:15]
            line_arm1[i], = ax.plot([c_x[i], r1_x[i]], [c_y[i], r1_y[i]], [c_z[i], r1_z[i]], linewidth=1, color='red',
                                    marker='o', markersize=2)
            line_arm2[i], = ax.plot([c_x[i], r2_x[i]], [c_y[i], r2_y[i]], [c_z[i], r2_z[i]], linewidth=1, color='blue',
                                    marker='o', markersize=2)
            line_arm3[i], = ax.plot([c_x[i], r3_x[i]], [c_y[i], r3_y[i]], [c_z[i], r3_z[i]], linewidth=1, color='red',
                                    marker='o', markersize=2)
            line_arm4[i], = ax.plot([c_x[i], r4_x[i]], [c_y[i], r4_y[i]], [c_z[i], r4_z[i]], linewidth=1, color='blue',
                                    marker='o', markersize=2)

            # line_traj_ref[i], = ax.plot(position_ref[:1, 0], position_ref[:1, 1], position_ref[:1, 2], color='gray', alpha=0.5)
            # c_x_ref, c_y_ref, c_z_ref = position_ref[0, 0:3]
            # r1_x_ref, r1_y_ref, r1_z_ref = position_ref[0, 3:6]
            # r2_x_ref, r2_y_ref, r2_z_ref = position_ref[0, 6:9]
            # r3_x_ref, r3_y_ref, r3_z_ref = position_ref[0, 9:12]
            # r4_x_ref, r4_y_ref, r4_z_ref = position_ref[0, 12:15]
            # line_arm1_ref, = ax.plot([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref], [c_z_ref, r1_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm2_ref, = ax.plot([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref], [c_z_ref, r2_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm3_ref, = ax.plot([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref], [c_z_ref, r3_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm4_ref, = ax.plot([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref], [c_z_ref, r4_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)

        # time label
        time_template = 'time = %.1fs'
        time_text = ax.text2D(0.66, 0.55, "time", transform=ax.transAxes)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))
        sim_horizon = np.size(position[0], 0)

        def update_traj(num):
            # customize
            time_text.set_text(time_template % (num * dt))

            # trajectory
            for i in range(self.num_uav):
                line_traj[i].set_data(position[i][:num, 0], position[i][:num, 1])
                line_traj[i].set_3d_properties(position[i][:num, 2])

                # uav
                c_x[i], c_y[i], c_z[i] = position[i][num, 0:3]
                r1_x[i], r1_y[i], r1_z[i] = position[i][num, 3:6]
                r2_x[i], r2_y[i], r2_z[i] = position[i][num, 6:9]
                r3_x[i], r3_y[i], r3_z[i] = position[i][num, 9:12]
                r4_x[i], r4_y[i], r4_z[i] = position[i][num, 12:15]

                line_arm1[i].set_data(np.array([[c_x[i], r1_x[i]], [c_y[i], r1_y[i]]]))
                line_arm1[i].set_3d_properties([c_z[i], r1_z[i]])

                line_arm2[i].set_data(np.array([[c_x[i], r2_x[i]], [c_y[i], r2_y[i]]]))
                line_arm2[i].set_3d_properties([c_z[i], r2_z[i]])

                line_arm3[i].set_data(np.array([[c_x[i], r3_x[i]], [c_y[i], r3_y[i]]]))
                line_arm3[i].set_3d_properties([c_z[i], r3_z[i]])

                line_arm4[i].set_data(np.array([[c_x[i], r4_x[i]], [c_y[i], r4_y[i]]]))
                line_arm4[i].set_3d_properties([c_z[i], r4_z[i]])

            # trajectory ref
            #     num = sim_horizon - 1
            #     line_traj_ref.set_data(position_ref[:num, 0], position_ref[:num, 1])
            #     line_traj_ref.set_3d_properties(position_ref[:num, 2])
            #
            #     # uav ref
            #     c_x_ref, c_y_ref, c_z_ref = position_ref[num, 0:3]
            #     r1_x_ref, r1_y_ref, r1_z_ref = position_ref[num, 3:6]
            #     r2_x_ref, r2_y_ref, r2_z_ref = position_ref[num, 6:9]
            #     r3_x_ref, r3_y_ref, r3_z_ref = position_ref[num, 9:12]
            #     r4_x_ref, r4_y_ref, r4_z_ref = position_ref[num, 12:15]
            #
            #     line_arm1_ref.set_data(np.array([[c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref]]))
            #     line_arm1_ref.set_3d_properties([c_z_ref, r1_z_ref])
            #
            #     line_arm2_ref.set_data(np.array([[c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref]]))
            #     line_arm2_ref.set_3d_properties([c_z_ref, r2_z_ref])
            #
            #     line_arm3_ref.set_data(np.array([[c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref]]))
            #     line_arm3_ref.set_3d_properties([c_z_ref, r3_z_ref])
            #
            #     line_arm4_ref.set_data(np.array([[c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref]]))
            #     line_arm4_ref.set_3d_properties([c_z_ref, r4_z_ref])

            # if num == sim_horizon-2:
            #    plt.pause(2)

            return line_traj[0], line_arm1[0], line_arm2[0], line_arm3[0], line_arm4[0], line_traj[1], line_arm1[1], \
                   line_arm2[1], line_arm3[1], line_arm4[1], line_traj[2], line_arm1[2], line_arm2[2], line_arm3[2], \
                   line_arm4[2], line_traj[3], line_arm1[3], line_arm2[3], line_arm3[3], line_arm4[3], time_text  # , \

            # line_traj_ref, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=100, blit=True)

        def restart():
            root = tkinter.Tk()
            root.withdraw()
            result = tkinter.messagebox.askyesno("Restart", "Do you want to restart animation?")
            if result:
                ani.frame_seq = ani.new_frame_seq()
                ani.event_source.start()
            else:
                plt.close()

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('case2' + title + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()

    def dir_cosine(self, q):
        C_B_I = vertcat(
            horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
        return C_B_I

    def skew(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2], v[1]),
            horzcat(v[2], 0, -v[0]),
            horzcat(-v[1], v[0], 0)
        )
        return v_cross

    def omega(self, w):
        omeg = vertcat(
            horzcat(0, -w[0], -w[1], -w[2]),
            horzcat(w[0], 0, w[2], -w[1]),
            horzcat(w[1], -w[2], 0, w[0]),
            horzcat(w[2], w[1], -w[0], 0)
        )
        return omeg

    def quaternion_mul(self, p, q):
        return vertcat(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                       p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                       p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                       p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
                       )


class UAV_formation_2D:
    def __init__(self, project_name='my UAV swarm 2D'):
        self.project_name = 'my uav swarm 2D'

        self.states_bd = [None, None]
        self.ctrl_bd = [None, None]
        self.UAV_labels = None
        self.obstacles = [[4.3, 3], [4.8, 3]]
        self.wall = [0, 5]

    def initDyn(self, uav_index=0, Jxy=None, mass=None, l=0.1, states_bd=[None, None],
                ctrl_bd=[None, None], dt=1e-3):
        # define the state of the uavs
        # Model reference : http://underactuated.mit.edu/acrobot.html
        rx, ry = SX.sym('rx' + str(uav_index)), SX.sym('ry' + str(uav_index))
        self.r_I = vertcat(rx, ry)
        vx, vy = SX.sym('vx' + str(uav_index)), SX.sym('vy' + str(uav_index))
        self.v_I = vertcat(vx, vy)

        self.theta = SX.sym('theta' + str(uav_index))
        self.w_B = SX.sym('w' + str(uav_index))
        # define the Planar quadrotor input
        f1, f2 = SX.sym('f1' + str(uav_index)), SX.sym('f2' + str(uav_index))
        self.T_B = vertcat(f1, f2)

        # global parameter
        g = 10

        # parameters settings
        parameter = []
        if Jxy is None:
            self.Jxy = SX.sym('Jxy' + str(uav_index))
            parameter += [self.Jxy]
        else:
            self.Jxy = Jxy

        if mass is None:
            self.mass = SX.sym('mass' + str(uav_index))
            parameter += [self.mass]
        else:
            self.mass = mass

        if l is None:
            self.l = SX.sym('l' + str(uav_index))
            parameter += [self.l]
        else:
            self.l = l

        self.dyn_auxvar = vcat(parameter)

        # Angular moment of inertia
        # self.J_B = diag(vertcat(self.Jx, self.Jy))
        # Gravity
        self.g_I = vertcat(0, -g)
        # Mass of rocket, assume is little changed during the landing process
        self.m = self.mass

        dr_I = self.v_I
        dv_I = 1 / self.m * vertcat(-sin(self.theta) * (self.T_B[0] + self.T_B[1]),
                                    cos(self.theta) * (self.T_B[0] + self.T_B[1])) + self.g_I
        dtheta = self.w_B
        dw_B = 1 / self.Jxy * self.l * (self.T_B[0] - self.T_B[1])

        self.states = vertcat(self.r_I, self.v_I, self.theta, self.w_B)

        if np.asarray(states_bd).size == self.states.numel():
            self.states_bd = states_bd
        else:
            self.states_bd = self.states.numel() * [-1e10, 1e10]

        self.ctrl = self.T_B
        if np.asarray(ctrl_bd).size == self.ctrl.numel():
            self.ctrl_bd = ctrl_bd
        else:
            self.states_bd = self.states.numel() * [-1e10, 1e10]

        self.dynf = vertcat(dr_I, dv_I, dtheta, dw_B)
        self.dt = dt

    def initCost(self, adj_matrix=None, uavswarm=None, num_uav=None, uav_index=None, w_F_r=None, w_F_v=None,
                 w_r_formation=None,
                 w_v_formation=None,
                 uav_dist=None, w_uav_collision=None, w_obst=None, wthrust=0.1):

        parameter = []
        if w_F_r is None:
            self.w_F_r = SX.sym('w_F_r' + str(uav_index))
            parameter += [self.w_F_r]
        else:
            self.w_F_r = w_F_r

        if w_F_v is None:
            self.w_F_v = SX.sym('w_F_v' + str(uav_index))
            parameter += [self.w_F_v]
        else:
            self.w_F_v = w_F_v

        if w_r_formation is None:
            self.w_r_formation = SX.sym('w_r_formation' + str(uav_index))
            parameter += [self.w_r_formation]
        else:
            self.w_r_formation = w_r_formation

        if w_v_formation is None:
            self.w_v_formation = SX.sym('w_v_formation' + str(uav_index))
            parameter += [self.w_v_formation]
        else:
            self.w_v_formation = w_v_formation

        if uav_dist is None:
            self.uav_dist = SX.sym('uav_dist' + str(uav_index))
            parameter += [self.uav_dist]
        else:
            self.uav_dist = uav_dist

        if w_uav_collision is None:
            self.w_uav_collision = SX.sym('w_uav_collision' + str(uav_index))
            parameter += [self.w_uav_collision]
        else:
            self.w_uav_collision = w_uav_collision

        if w_obst is None:
            self.w_obst = SX.sym('w_obst' + str(uav_index))
            parameter += [self.w_obst]
        else:
            self.w_obst = w_obst

        self.cost_auxvar = vcat(parameter)

        # neighbors

        # formation distance and speed costs
        if adj_matrix:
            neighbors = list(np.where(np.array(adj_matrix) == 1)[0])
        else:
            neighbors = [*range(num_uav)]
        neighbors.remove(uav_index)

        self.cost_r_formation = 0
        self.cost_v_formation = 0
        self.cost_v_formation += dot(self.v_I[1] - 2, self.v_I[1] - 2)

        for j in neighbors:
            self.cost_r_formation += dot(self.r_I - uavswarm[j].r_I - vcat([(uav_index - j) * self.uav_dist, 0]),
                                         self.r_I - uavswarm[j].r_I - vcat([(uav_index - j) * self.uav_dist, 0]))
            self.cost_v_formation += dot(self.v_I - uavswarm[j].v_I, self.v_I - uavswarm[j].v_I)

        # if uav_index == 1:
        #    self.cost_r_formation += dot(self.r_I - uavswarm[3].r_I - vcat([(-2) * self.uav_dist, 0]),
        #                                 self.r_I - uavswarm[3].r_I - vcat([(-2) * self.uav_dist, 0]))
        #    self.cost_v_formation += dot(self.v_I - uavswarm[3].v_I, self.v_I - uavswarm[3].v_I)
        # if uav_index == 3:
        #    self.cost_r_formation += dot(self.r_I - uavswarm[1].r_I - vcat([(2) * self.uav_dist, 0]),
        #                                 self.r_I - uavswarm[1].r_I - vcat([(2) * self.uav_dist, 0]))
        #    self.cost_v_formation += dot(self.v_I - uavswarm[1].v_I, self.v_I - uavswarm[1].v_I)
        # for i in neighbors:
        #    self.cost_v_formation = self.cost_v_formation + norm_2(self.v_I - uavswarm[i].v_I)

        # goal_height = 5
        # self.cost_altitude = dot(self.r_I[2] - goal_height, self.r_I[2] - goal_height)

        # if uav_index == 0:
        goal_r_I = np.array([1, 7])
        self.cost_r_I = dot(self.r_I - goal_r_I - vcat([uav_index * self.uav_dist, 0]),
                            self.r_I - goal_r_I - vcat([uav_index * self.uav_dist, 0]))

        # else:
        #    self.cost_r_I =0

        #    self.cost_r_I = 0
        # goal velocity cost
        goal_v_I = np.array([0, 2])
        self.cost_v_I = dot(self.v_I - goal_v_I, self.v_I - goal_v_I)

        # collision cost
        self.cost_collision = 0
        for i in neighbors:
            # self.cost_collision = self.cost_collision + 1 / fmin(fmax(norm_2(self.r_I - uavswarm[i].r_I), 1e-3),
            #                                                     0.6) - 1 / 0.6
            self.cost_collision = self.cost_collision + 1 / dot(self.r_I - uavswarm[i].r_I, self.r_I - uavswarm[i].r_I)

        # self.cost_obst = 5 * fmax(self.wall[0] - self.r_I[0], 0) + 5 * fmax(self.r_I[0] - self.wall[1], 0) + \
        #                 1 / (fmin(fmax(norm_2(self.r_I[0:2] - self.obstacles[0]), 1e-6), 0.6)) - 1 / 0.6 + \
        #                 1 / (fmin(fmax(norm_2(self.r_I[0:2] - self.obstacles[1]), 1e-6), 0.6)) - 1 / 0.6

        self.cost_obst = 1 / dot(self.r_I[0:2] - self.obstacles[0], self.r_I[0:2] - self.obstacles[0]) + \
                         1 / dot(self.r_I[0:2] - self.obstacles[1], self.r_I[0:2] - self.obstacles[1])
        # the thrust cost
        self.cost_thrust = dot(self.T_B, self.T_B)

        self.path_cost = self.w_r_formation * self.cost_r_formation + \
                         self.w_v_formation * self.cost_v_formation + \
                         self.w_uav_collision * self.cost_collision + \
                         self.w_obst * self.cost_obst + \
                         wthrust * self.cost_thrust
        # self.w_v_formation * self.cost_v_formation + \
        # self.w_F_r * self.cost_r_I + \
        # self.w_F_v * self.cost_v_I + \

        self.final_cost = self.w_F_r * self.cost_r_I + \
                          self.w_F_v * self.cost_v_I + \
                          10 * self.w_r_formation * self.cost_r_formation + \
                          10 * self.w_v_formation * self.cost_v_formation + \
                          self.w_uav_collision * self.cost_collision

    def get_UAV_position(self, wing_len, state_traj):

        # thrust_position in body frame

        # horizon
        horizon = np.size(state_traj, 0)
        position = np.zeros((horizon, 6))
        for t in range(horizon):
            # position of COM
            rc = state_traj[t, 0:2]
            # altitude of quaternion
            theta = state_traj[t, 4]

            # position of each rotor in inertial frame
            r1_pos = rc + vertcat(-cos(theta) * wing_len / 2, -sin(theta) * wing_len / 2).T
            r2_pos = rc + vertcat(cos(theta) * wing_len / 2, sin(theta) * wing_len / 2).T

            # store
            position[t, 0:2] = rc
            position[t, 2:4] = r1_pos
            position[t, 4:6] = r2_pos

        return position

    def play_multi_animation(self, uavswarm, wing_len, state_traj, state_traj_ref=None, dt=None, save_option=0,
                             title='UAV Maneuvering'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_ylim(-1, 10)
        ax.set_xlim(-1, 6)
        ax.set_box_aspect(1)
        ax.set_title(title, pad=20, fontsize=15)

        wall1 = plt.Rectangle((0, -1), 0.1, 11, color='limegreen', alpha=0.2)
        wall2 = plt.Rectangle((5, -1), 0.1, 11, color='limegreen', alpha=0.2)
        risky = plt.Rectangle((self.obstacles[0][0] - 0.5, self.obstacles[0][1] - 0.8), 1.5, 1, color='r', alpha=0.5)

        ax.add_patch(wall1)
        ax.add_patch(wall2)
        ax.add_patch(risky)

        # data
        self.num_uav = len(uavswarm)
        position = [None] * self.num_uav
        position_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            position[i] = self.get_UAV_position(wing_len, state_traj[i])
            if state_traj_ref is None:
                position_ref[i] = self.get_UAV_position(0, numpy.zeros_like(position[i]))
            else:
                position_ref[i] = self.get_UAV_position(wing_len, state_traj_ref[i])

        # animation
        line_traj = [None] * self.num_uav
        c_x = [None] * self.num_uav
        c_y = [None] * self.num_uav

        r1_x = [None] * self.num_uav
        r1_y = [None] * self.num_uav

        r2_x = [None] * self.num_uav
        r2_y = [None] * self.num_uav

        line_arm1 = [None] * self.num_uav
        line_arm2 = [None] * self.num_uav

        line_traj_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            line_traj[i], = ax.plot(position[i][:1, 0], position[i][:1, 1])
            c_x[i], c_y[i] = position[i][0, 0:2]
            r1_x[i], r1_y[i] = position[i][0, 2:4]
            r2_x[i], r2_y[i] = position[i][0, 4:6]

            line_arm1[i], = ax.plot([c_x[i], r1_x[i]], [c_y[i], r1_y[i]], linewidth=1, color='red',
                                    marker='o', markersize=2)
            line_arm2[i], = ax.plot([c_x[i], r2_x[i]], [c_y[i], r2_y[i]], linewidth=1, color='blue',
                                    marker='o', markersize=2)

            # line_traj_ref[i], = ax.plot(position_ref[:1, 0], position_ref[:1, 1], position_ref[:1, 2], color='gray', alpha=0.5)
            # c_x_ref, c_y_ref, c_z_ref = position_ref[0, 0:3]
            # r1_x_ref, r1_y_ref, r1_z_ref = position_ref[0, 3:6]
            # r2_x_ref, r2_y_ref, r2_z_ref = position_ref[0, 6:9]
            # r3_x_ref, r3_y_ref, r3_z_ref = position_ref[0, 9:12]
            # r4_x_ref, r4_y_ref, r4_z_ref = position_ref[0, 12:15]
            # line_arm1_ref, = ax.plot([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref], [c_z_ref, r1_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm2_ref, = ax.plot([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref], [c_z_ref, r2_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm3_ref, = ax.plot([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref], [c_z_ref, r3_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm4_ref, = ax.plot([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref], [c_z_ref, r4_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)

        # time label
        time_template = 'time = %.1fs'
        time_text = ax.text(0.66, 0.55, "time", transform=ax.transAxes)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))
        sim_horizon = np.size(position[0], 0)

        def update_traj(num):
            # customize
            time_text.set_text(time_template % (num * dt))

            # trajectory
            for i in range(self.num_uav):
                line_traj[i].set_data(position[i][:num, 0], position[i][:num, 1])

                # uav
                c_x[i], c_y[i] = position[i][num, 0:2]
                r1_x[i], r1_y[i] = position[i][num, 2:4]
                r2_x[i], r2_y[i] = position[i][num, 4:6]

                line_arm1[i].set_data(np.array([[c_x[i], r1_x[i]], [c_y[i], r1_y[i]]]))

                line_arm2[i].set_data(np.array([[c_x[i], r2_x[i]], [c_y[i], r2_y[i]]]))

            # trajectory ref
            #     num = sim_horizon - 1
            #     line_traj_ref.set_data(position_ref[:num, 0], position_ref[:num, 1])
            #     line_traj_ref.set_3d_properties(position_ref[:num, 2])
            #
            #     # uav ref
            #     c_x_ref, c_y_ref, c_z_ref = position_ref[num, 0:3]
            #     r1_x_ref, r1_y_ref, r1_z_ref = position_ref[num, 3:6]
            #     r2_x_ref, r2_y_ref, r2_z_ref = position_ref[num, 6:9]
            #     r3_x_ref, r3_y_ref, r3_z_ref = position_ref[num, 9:12]
            #     r4_x_ref, r4_y_ref, r4_z_ref = position_ref[num, 12:15]
            #
            #     line_arm1_ref.set_data(np.array([[c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref]]))
            #     line_arm1_ref.set_3d_properties([c_z_ref, r1_z_ref])
            #
            #     line_arm2_ref.set_data(np.array([[c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref]]))
            #     line_arm2_ref.set_3d_properties([c_z_ref, r2_z_ref])
            #
            #     line_arm3_ref.set_data(np.array([[c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref]]))
            #     line_arm3_ref.set_3d_properties([c_z_ref, r3_z_ref])
            #
            #     line_arm4_ref.set_data(np.array([[c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref]]))
            #     line_arm4_ref.set_3d_properties([c_z_ref, r4_z_ref])

            # if num == sim_horizon-2:
            #    plt.pause(2)

            return line_traj[0], line_arm1[0], line_arm2[0], line_traj[1], line_arm1[1], \
                   line_arm2[1], line_traj[2], line_arm1[2], line_arm2[2], line_traj[3], line_arm1[3], line_arm2[
                       3], time_text  # , \

            # line_traj_ref, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=100, blit=True)

        def restart():
            root = tkinter.Tk()
            root.withdraw()
            result = tkinter.messagebox.askyesno("Restart", "Do you want to restart animation?")
            if result:
                ani.frame_seq = ani.new_frame_seq()
                ani.event_source.start()
            else:
                plt.close()

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('case2' + title + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()

    def play_multi_animation_3(self, uavswarm, wing_len, state_traj, state_traj_ref=None, dt=None, save_option=1,
                               title='UAV Maneuvering'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_ylim(-1, 10)
        ax.set_xlim(-1, 6)
        ax.set_box_aspect(1)
        ax.set_title(title, pad=20, fontsize=15)

        wall1 = plt.Rectangle((0, -1), 0.1, 11, color='limegreen', alpha=0.2)
        wall2 = plt.Rectangle((5, -1), 0.1, 11, color='limegreen', alpha=0.2)
        risky = plt.Rectangle((self.obstacles[0][0] - 0.5, self.obstacles[0][1] - 0.8), 1.2, 1, color='r', alpha=0.5)

        ax.add_patch(wall1)
        ax.add_patch(wall2)
        ax.add_patch(risky)

        # data
        self.num_uav = len(uavswarm)
        position = [None] * self.num_uav
        position_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            position[i] = self.get_UAV_position(wing_len, state_traj[i])
            if state_traj_ref is None:
                position_ref[i] = self.get_UAV_position(0, numpy.zeros_like(position[i]))
            else:
                position_ref[i] = self.get_UAV_position(wing_len, state_traj_ref[i])

        # animation
        line_traj = [None] * self.num_uav
        c_x = [None] * self.num_uav
        c_y = [None] * self.num_uav

        r1_x = [None] * self.num_uav
        r1_y = [None] * self.num_uav

        r2_x = [None] * self.num_uav
        r2_y = [None] * self.num_uav

        line_arm1 = [None] * self.num_uav
        line_arm2 = [None] * self.num_uav

        line_traj_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            line_traj[i], = ax.plot(position[i][:1, 0], position[i][:1, 1])
            c_x[i], c_y[i] = position[i][0, 0:2]
            r1_x[i], r1_y[i] = position[i][0, 2:4]
            r2_x[i], r2_y[i] = position[i][0, 4:6]

            line_arm1[i], = ax.plot([c_x[i], r1_x[i]], [c_y[i], r1_y[i]], linewidth=1, color='red',
                                    marker='o', markersize=2)
            line_arm2[i], = ax.plot([c_x[i], r2_x[i]], [c_y[i], r2_y[i]], linewidth=1, color='blue',
                                    marker='o', markersize=2)

            # line_traj_ref[i], = ax.plot(position_ref[:1, 0], position_ref[:1, 1], position_ref[:1, 2], color='gray', alpha=0.5)
            # c_x_ref, c_y_ref, c_z_ref = position_ref[0, 0:3]
            # r1_x_ref, r1_y_ref, r1_z_ref = position_ref[0, 3:6]
            # r2_x_ref, r2_y_ref, r2_z_ref = position_ref[0, 6:9]
            # r3_x_ref, r3_y_ref, r3_z_ref = position_ref[0, 9:12]
            # r4_x_ref, r4_y_ref, r4_z_ref = position_ref[0, 12:15]
            # line_arm1_ref, = ax.plot([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref], [c_z_ref, r1_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm2_ref, = ax.plot([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref], [c_z_ref, r2_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm3_ref, = ax.plot([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref], [c_z_ref, r3_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm4_ref, = ax.plot([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref], [c_z_ref, r4_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)

        # time label
        time_template = 'time = %.1fs'
        time_text = ax.text(0.66, 0.55, "time", transform=ax.transAxes)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))
        sim_horizon = np.size(position[0], 0)

        def update_traj(num):
            # customize
            time_text.set_text(time_template % ((num+1) *2 * dt))

            # trajectory
            for i in range(self.num_uav):
                line_traj[i].set_data(position[i][:num, 0], position[i][:num, 1])

                # uav
                c_x[i], c_y[i] = position[i][num, 0:2]
                r1_x[i], r1_y[i] = position[i][num, 2:4]
                r2_x[i], r2_y[i] = position[i][num, 4:6]

                line_arm1[i].set_data(np.array([[c_x[i], r1_x[i]], [c_y[i], r1_y[i]]]))

                line_arm2[i].set_data(np.array([[c_x[i], r2_x[i]], [c_y[i], r2_y[i]]]))

            # trajectory ref
            #     num = sim_horizon - 1
            #     line_traj_ref.set_data(position_ref[:num, 0], position_ref[:num, 1])
            #     line_traj_ref.set_3d_properties(position_ref[:num, 2])
            #
            #     # uav ref
            #     c_x_ref, c_y_ref, c_z_ref = position_ref[num, 0:3]
            #     r1_x_ref, r1_y_ref, r1_z_ref = position_ref[num, 3:6]
            #     r2_x_ref, r2_y_ref, r2_z_ref = position_ref[num, 6:9]
            #     r3_x_ref, r3_y_ref, r3_z_ref = position_ref[num, 9:12]
            #     r4_x_ref, r4_y_ref, r4_z_ref = position_ref[num, 12:15]
            #
            #     line_arm1_ref.set_data(np.array([[c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref]]))
            #     line_arm1_ref.set_3d_properties([c_z_ref, r1_z_ref])
            #
            #     line_arm2_ref.set_data(np.array([[c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref]]))
            #     line_arm2_ref.set_3d_properties([c_z_ref, r2_z_ref])
            #
            #     line_arm3_ref.set_data(np.array([[c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref]]))
            #     line_arm3_ref.set_3d_properties([c_z_ref, r3_z_ref])
            #
            #     line_arm4_ref.set_data(np.array([[c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref]]))
            #     line_arm4_ref.set_3d_properties([c_z_ref, r4_z_ref])

            # if num == sim_horizon-2:
            #    plt.pause(2)

            return line_traj[0], line_arm1[0], line_arm2[0], line_traj[1], line_arm1[1], \
                   line_arm2[1], line_traj[2], line_arm1[2], line_arm2[2], line_traj[3], line_arm1[3], line_arm2[
                       3], time_text  # , \

            # line_traj_ref, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=200, blit=True)

        def restart():
            root = tkinter.Tk()
            root.withdraw()
            result = tkinter.messagebox.askyesno("Restart", "Do you want to restart animation?")
            if result:
                ani.frame_seq = ani.new_frame_seq()
                ani.event_source.start()
            else:
                plt.close()

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('case2' + title + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()

    def dir_cosine(self, q):
        C_B_I = vertcat(
            horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
        return C_B_I

    def skew(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2], v[1]),
            horzcat(v[2], 0, -v[0]),
            horzcat(-v[1], v[0], 0)
        )
        return v_cross

    def omega(self, w):
        omeg = vertcat(
            horzcat(0, -w[0], -w[1], -w[2]),
            horzcat(w[0], 0, w[2], -w[1]),
            horzcat(w[1], -w[2], 0, w[0]),
            horzcat(w[2], w[1], -w[0], 0)
        )
        return omeg

    def quaternion_mul(self, p, q):
        return vertcat(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                       p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                       p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                       p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
                       )


class UAV_formation_diamond_2D:
    def __init__(self, project_name='my UAV swarm 2D'):
        self.project_name = 'my uav swarm 2D'

        self.states_bd = [None, None]
        self.ctrl_bd = [None, None]
        self.UAV_labels = None
        self.obstacles = [[4.3, 3], [4.8, 3]]
        self.wall = [0, 5]

    def initDyn(self, uav_index=0, Jxy=None, mass=None, l=0.1, states_bd=[None, None],
                ctrl_bd=[None, None], dt=1e-3):
        # define the state of the uavs
        # Model reference : http://underactuated.mit.edu/acrobot.html
        rx, ry = SX.sym('rx' + str(uav_index)), SX.sym('ry' + str(uav_index))
        self.r_I = vertcat(rx, ry)
        vx, vy = SX.sym('vx' + str(uav_index)), SX.sym('vy' + str(uav_index))
        self.v_I = vertcat(vx, vy)

        self.theta = SX.sym('theta' + str(uav_index))
        self.w_B = SX.sym('w' + str(uav_index))
        # define the Planar quadrotor input
        f1, f2 = SX.sym('f1' + str(uav_index)), SX.sym('f2' + str(uav_index))
        self.T_B = vertcat(f1, f2)

        # global parameter
        g = 10

        # parameters settings
        parameter = []
        if Jxy is None:
            self.Jxy = SX.sym('Jxy' + str(uav_index))
            parameter += [self.Jxy]
        else:
            self.Jxy = Jxy

        if mass is None:
            self.mass = SX.sym('mass' + str(uav_index))
            parameter += [self.mass]
        else:
            self.mass = mass

        if l is None:
            self.l = SX.sym('l' + str(uav_index))
            parameter += [self.l]
        else:
            self.l = l

        self.dyn_auxvar = vcat(parameter)

        # Angular moment of inertia
        # self.J_B = diag(vertcat(self.Jx, self.Jy))
        # Gravity
        self.g_I = vertcat(0, -g)
        # Mass of rocket, assume is little changed during the landing process
        self.m = self.mass

        dr_I = self.v_I
        dv_I = 1 / self.m * vertcat(-sin(self.theta) * (self.T_B[0] + self.T_B[1]),
                                    cos(self.theta) * (self.T_B[0] + self.T_B[1])) + self.g_I
        dtheta = self.w_B
        dw_B = 1 / self.Jxy * self.l * (self.T_B[0] - self.T_B[1])

        self.states = vertcat(self.r_I, self.v_I, self.theta, self.w_B)

        if np.asarray(states_bd).size == self.states.numel():
            self.states_bd = states_bd
        else:
            self.states_bd = self.states.numel() * [-1e10, 1e10]

        self.ctrl = self.T_B
        if np.asarray(ctrl_bd).size == self.ctrl.numel():
            self.ctrl_bd = ctrl_bd
        else:
            self.states_bd = self.states.numel() * [-1e10, 1e10]

        self.dynf = vertcat(dr_I, dv_I, dtheta, dw_B)
        self.dt = dt

    def initCost(self, adj_matrix=None, uavswarm=None, num_uav=None, uav_index=None, w_F_r=None, w_F_v=None,
                 w_r_formation=None,
                 w_v_formation=None,
                 uav_dist=None, w_uav_collision=None, w_obst=None, wthrust=0.1):

        parameter = []
        if w_F_r is None:
            self.w_F_r = SX.sym('w_F_r' + str(uav_index))
            parameter += [self.w_F_r]
        else:
            self.w_F_r = w_F_r

        if w_F_v is None:
            self.w_F_v = SX.sym('w_F_v' + str(uav_index))
            parameter += [self.w_F_v]
        else:
            self.w_F_v = w_F_v

        if w_r_formation is None:
            self.w_r_formation = SX.sym('w_r_formation' + str(uav_index))
            parameter += [self.w_r_formation]
        else:
            self.w_r_formation = w_r_formation

        if w_v_formation is None:
            self.w_v_formation = SX.sym('w_v_formation' + str(uav_index))
            parameter += [self.w_v_formation]
        else:
            self.w_v_formation = w_v_formation

        if uav_dist is None:
            self.uav_dist = SX.sym('uav_dist' + str(uav_index))
            parameter += [self.uav_dist]
        else:
            self.uav_dist = uav_dist

        if w_uav_collision is None:
            self.w_uav_collision = SX.sym('w_uav_collision' + str(uav_index))
            parameter += [self.w_uav_collision]
        else:
            self.w_uav_collision = w_uav_collision

        if w_obst is None:
            self.w_obst = SX.sym('w_obst' + str(uav_index))
            parameter += [self.w_obst]
        else:
            self.w_obst = w_obst

        self.cost_auxvar = vcat(parameter)

        # neighbors

        # formation distance and speed costs
        if adj_matrix:
            neighbors = list(np.where(np.array(adj_matrix) == 1)[0])
        else:
            neighbors = [*range(num_uav)]
        neighbors.remove(uav_index)

        self.cost_r_formation = 0
        self.cost_v_formation = 0
        self.cost_v_formation += dot(self.v_I[1] - 2, self.v_I[1] - 2)

        if uav_index == 0:
            self.cost_r_formation += dot(self.r_I[1] - uavswarm[2].r_I[1],self.r_I[1] - uavswarm[2].r_I[1])
            self.cost_r_formation += dot(dot(self.r_I - uavswarm[1].r_I, self.r_I - uavswarm[1].r_I) - self.uav_dist**2,
                                         dot(self.r_I - uavswarm[1].r_I, self.r_I - uavswarm[1].r_I) - self.uav_dist**2)
            self.cost_r_formation += dot(dot(self.r_I - uavswarm[3].r_I, self.r_I - uavswarm[3].r_I) - self.uav_dist**2,
                                         dot(self.r_I - uavswarm[3].r_I, self.r_I - uavswarm[3].r_I) - self.uav_dist**2)
        elif uav_index == 1:
            self.cost_r_formation += dot(self.r_I - uavswarm[2].r_I - vcat([-0.866 * self.uav_dist, 0.5 * self.uav_dist]),
                                         self.r_I - uavswarm[2].r_I - vcat([-0.866 * self.uav_dist, 0.5 * self.uav_dist]))
        elif uav_index == 2:
            self.cost_r_formation += dot(self.r_I - uavswarm[1].r_I - vcat([0.866 * self.uav_dist, -0.5 * self.uav_dist]),
                                         self.r_I - uavswarm[1].r_I - vcat([0.866 * self.uav_dist, -0.5 * self.uav_dist]))
        elif uav_index == 3:
            self.cost_r_formation += dot(self.r_I[0] - uavswarm[1].r_I[0],self.r_I[0] - uavswarm[1].r_I[0])
            self.cost_r_formation += dot(dot(self.r_I - uavswarm[1].r_I, self.r_I - uavswarm[1].r_I) - self.uav_dist**2,
                                         dot(self.r_I - uavswarm[1].r_I, self.r_I - uavswarm[1].r_I) - self.uav_dist**2)
            self.cost_r_formation += dot(dot(self.r_I - uavswarm[2].r_I, self.r_I - uavswarm[2].r_I) - self.uav_dist**2,
                                         dot(self.r_I - uavswarm[2].r_I, self.r_I - uavswarm[2].r_I) - self.uav_dist**2)

        for j in neighbors:
            self.cost_v_formation += dot(self.v_I - uavswarm[j].v_I, self.v_I - uavswarm[j].v_I)


        goal_r1_I = np.array([2.5, 8])
        goal_r2_I = np.array([4.232, 7])

        if uav_index == 0:
            self.cost_r_I = dot(dot(self.r_I - uavswarm[1].r_I, self.r_I - uavswarm[1].r_I) - self.uav_dist**2,
                                dot(self.r_I - uavswarm[1].r_I, self.r_I - uavswarm[1].r_I) - self.uav_dist**2)
            self.cost_r_I += dot(dot(self.r_I - uavswarm[3].r_I, self.r_I - uavswarm[3].r_I) - self.uav_dist ** 2,
                                dot(self.r_I - uavswarm[3].r_I, self.r_I - uavswarm[3].r_I) - self.uav_dist ** 2)
        elif uav_index == 1:
            self.cost_r_I = dot(self.r_I - goal_r1_I, self.r_I - goal_r1_I)

        elif uav_index == 2:
            self.cost_r_I = dot(self.r_I - goal_r2_I, self.r_I - goal_r2_I)

        elif uav_index == 3:
            self.cost_r_I = dot(dot(self.r_I - goal_r1_I, self.r_I - goal_r1_I) - self.uav_dist ** 2,
                                dot(self.r_I - goal_r1_I, self.r_I - goal_r1_I) - self.uav_dist ** 2)
            self.cost_r_I += dot(dot(self.r_I - goal_r2_I, self.r_I - goal_r2_I) - self.uav_dist ** 2 ,
                                dot(self.r_I - goal_r2_I, self.r_I - goal_r2_I) - self.uav_dist ** 2 )


        # else:
        #    self.cost_r_I =0

        #    self.cost_r_I = 0
        # goal velocity cost
        goal_v_I = np.array([0, 2])
        self.cost_v_I = dot(self.v_I - goal_v_I, self.v_I - goal_v_I)

        # collision cost
        self.cost_collision = 0
        for i in neighbors:
            # self.cost_collision = self.cost_collision + 1 / fmin(fmax(norm_2(self.r_I - uavswarm[i].r_I), 1e-3),
            #                                                     0.6) - 1 / 0.6
            self.cost_collision = self.cost_collision + 1 / dot(self.r_I - uavswarm[i].r_I, self.r_I - uavswarm[i].r_I)

        # self.cost_obst = 5 * fmax(self.wall[0] - self.r_I[0], 0) + 5 * fmax(self.r_I[0] - self.wall[1], 0) + \
        #                 1 / (fmin(fmax(norm_2(self.r_I[0:2] - self.obstacles[0]), 1e-6), 0.6)) - 1 / 0.6 + \
        #                 1 / (fmin(fmax(norm_2(self.r_I[0:2] - self.obstacles[1]), 1e-6), 0.6)) - 1 / 0.6

        self.cost_obst = 1 / dot(self.r_I[0:2] - self.obstacles[0], self.r_I[0:2] - self.obstacles[0]) + \
                         1 / dot(self.r_I[0:2] - self.obstacles[1], self.r_I[0:2] - self.obstacles[1])
        # the thrust cost
        self.cost_thrust = dot(self.T_B, self.T_B)

        self.path_cost = self.w_r_formation * self.cost_r_formation + \
                         self.w_v_formation * self.cost_v_formation + \
                         self.w_uav_collision * self.cost_collision + \
                         self.w_obst * self.cost_obst + \
                         wthrust * self.cost_thrust
        # self.w_v_formation * self.cost_v_formation + \
        # self.w_F_r * self.cost_r_I + \
        # self.w_F_v * self.cost_v_I + \

        self.final_cost = self.w_F_r * self.cost_r_I + \
                          self.w_F_v * self.cost_v_I + \
                          10 * self.w_r_formation * self.cost_r_formation + \
                          10 * self.w_v_formation * self.cost_v_formation + \
                          self.w_uav_collision * self.cost_collision

    def get_UAV_position(self, wing_len, state_traj):

        # thrust_position in body frame

        # horizon
        horizon = np.size(state_traj, 0)
        position = np.zeros((horizon, 6))
        for t in range(horizon):
            # position of COM
            rc = state_traj[t, 0:2]
            # altitude of quaternion
            theta = state_traj[t, 4]

            # position of each rotor in inertial frame
            r1_pos = rc + vertcat(-cos(theta) * wing_len / 2, -sin(theta) * wing_len / 2).T
            r2_pos = rc + vertcat(cos(theta) * wing_len / 2, sin(theta) * wing_len / 2).T

            # store
            position[t, 0:2] = rc
            position[t, 2:4] = r1_pos
            position[t, 4:6] = r2_pos

        return position

    def play_multi_animation(self, uavswarm, wing_len, state_traj, state_traj_ref=None, dt=None, save_option=0,
                             title='UAV Maneuvering'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_ylim(-1, 10)
        ax.set_xlim(-1, 6)
        ax.set_box_aspect(1)
        ax.set_title(title, pad=20, fontsize=15)

        wall1 = plt.Rectangle((0, -1), 0.1, 11, color='limegreen', alpha=0.2)
        wall2 = plt.Rectangle((5, -1), 0.1, 11, color='limegreen', alpha=0.2)
        risky = plt.Rectangle((self.obstacles[0][0] - 0.5, self.obstacles[0][1] - 0.8), 1.5, 1, color='r', alpha=0.5)

        ax.add_patch(wall1)
        ax.add_patch(wall2)
        ax.add_patch(risky)

        # data
        self.num_uav = len(uavswarm)
        position = [None] * self.num_uav
        position_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            position[i] = self.get_UAV_position(wing_len, state_traj[i])
            if state_traj_ref is None:
                position_ref[i] = self.get_UAV_position(0, numpy.zeros_like(position[i]))
            else:
                position_ref[i] = self.get_UAV_position(wing_len, state_traj_ref[i])

        # animation
        line_traj = [None] * self.num_uav
        c_x = [None] * self.num_uav
        c_y = [None] * self.num_uav

        r1_x = [None] * self.num_uav
        r1_y = [None] * self.num_uav

        r2_x = [None] * self.num_uav
        r2_y = [None] * self.num_uav

        line_arm1 = [None] * self.num_uav
        line_arm2 = [None] * self.num_uav

        line_traj_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            line_traj[i], = ax.plot(position[i][:1, 0], position[i][:1, 1])
            c_x[i], c_y[i] = position[i][0, 0:2]
            r1_x[i], r1_y[i] = position[i][0, 2:4]
            r2_x[i], r2_y[i] = position[i][0, 4:6]

            line_arm1[i], = ax.plot([c_x[i], r1_x[i]], [c_y[i], r1_y[i]], linewidth=1, color='red',
                                    marker='o', markersize=2)
            line_arm2[i], = ax.plot([c_x[i], r2_x[i]], [c_y[i], r2_y[i]], linewidth=1, color='blue',
                                    marker='o', markersize=2)

            # line_traj_ref[i], = ax.plot(position_ref[:1, 0], position_ref[:1, 1], position_ref[:1, 2], color='gray', alpha=0.5)
            # c_x_ref, c_y_ref, c_z_ref = position_ref[0, 0:3]
            # r1_x_ref, r1_y_ref, r1_z_ref = position_ref[0, 3:6]
            # r2_x_ref, r2_y_ref, r2_z_ref = position_ref[0, 6:9]
            # r3_x_ref, r3_y_ref, r3_z_ref = position_ref[0, 9:12]
            # r4_x_ref, r4_y_ref, r4_z_ref = position_ref[0, 12:15]
            # line_arm1_ref, = ax.plot([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref], [c_z_ref, r1_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm2_ref, = ax.plot([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref], [c_z_ref, r2_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm3_ref, = ax.plot([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref], [c_z_ref, r3_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm4_ref, = ax.plot([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref], [c_z_ref, r4_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)

        # time label
        time_template = 'time = %.1fs'
        time_text = ax.text(0.66, 0.55, "time", transform=ax.transAxes)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))
        sim_horizon = np.size(position[0], 0)

        def update_traj(num):
            # customize
            time_text.set_text(time_template % (num * dt))

            # trajectory
            for i in range(self.num_uav):
                line_traj[i].set_data(position[i][:num, 0], position[i][:num, 1])

                # uav
                c_x[i], c_y[i] = position[i][num, 0:2]
                r1_x[i], r1_y[i] = position[i][num, 2:4]
                r2_x[i], r2_y[i] = position[i][num, 4:6]

                line_arm1[i].set_data(np.array([[c_x[i], r1_x[i]], [c_y[i], r1_y[i]]]))

                line_arm2[i].set_data(np.array([[c_x[i], r2_x[i]], [c_y[i], r2_y[i]]]))

            # trajectory ref
            #     num = sim_horizon - 1
            #     line_traj_ref.set_data(position_ref[:num, 0], position_ref[:num, 1])
            #     line_traj_ref.set_3d_properties(position_ref[:num, 2])
            #
            #     # uav ref
            #     c_x_ref, c_y_ref, c_z_ref = position_ref[num, 0:3]
            #     r1_x_ref, r1_y_ref, r1_z_ref = position_ref[num, 3:6]
            #     r2_x_ref, r2_y_ref, r2_z_ref = position_ref[num, 6:9]
            #     r3_x_ref, r3_y_ref, r3_z_ref = position_ref[num, 9:12]
            #     r4_x_ref, r4_y_ref, r4_z_ref = position_ref[num, 12:15]
            #
            #     line_arm1_ref.set_data(np.array([[c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref]]))
            #     line_arm1_ref.set_3d_properties([c_z_ref, r1_z_ref])
            #
            #     line_arm2_ref.set_data(np.array([[c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref]]))
            #     line_arm2_ref.set_3d_properties([c_z_ref, r2_z_ref])
            #
            #     line_arm3_ref.set_data(np.array([[c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref]]))
            #     line_arm3_ref.set_3d_properties([c_z_ref, r3_z_ref])
            #
            #     line_arm4_ref.set_data(np.array([[c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref]]))
            #     line_arm4_ref.set_3d_properties([c_z_ref, r4_z_ref])

            # if num == sim_horizon-2:
            #    plt.pause(2)

            return line_traj[0], line_arm1[0], line_arm2[0], line_traj[1], line_arm1[1], \
                   line_arm2[1], line_traj[2], line_arm1[2], line_arm2[2], line_traj[3], line_arm1[3], line_arm2[
                       3], time_text  # , \

            # line_traj_ref, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=100, blit=True)

        def restart():
            root = tkinter.Tk()
            root.withdraw()
            result = tkinter.messagebox.askyesno("Restart", "Do you want to restart animation?")
            if result:
                ani.frame_seq = ani.new_frame_seq()
                ani.event_source.start()
            else:
                plt.close()

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('case2' + title + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()


    def play_multi_animation_3(self, uavswarm, wing_len, state_traj, state_traj_ref=None, dt=None, save_option=0,
                               title='UAV Maneuvering'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_ylim(-1, 10)
        ax.set_xlim(-1, 6)
        ax.set_box_aspect(1)
        ax.set_title(title, pad=20, fontsize=15)

        wall1 = plt.Rectangle((0, -1), 0.1, 11, color='limegreen', alpha=0.2)
        wall2 = plt.Rectangle((5, -1), 0.1, 11, color='limegreen', alpha=0.2)
        risky = plt.Rectangle((self.obstacles[0][0] - 0.5, self.obstacles[0][1] - 0.8), 1.2, 1, color='r', alpha=0.5)

        ax.add_patch(wall1)
        ax.add_patch(wall2)
        ax.add_patch(risky)

        # data
        self.num_uav = len(uavswarm)
        position = [None] * self.num_uav
        position_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            position[i] = self.get_UAV_position(wing_len, state_traj[i])
            if state_traj_ref is None:
                position_ref[i] = self.get_UAV_position(0, numpy.zeros_like(position[i]))
            else:
                position_ref[i] = self.get_UAV_position(wing_len, state_traj_ref[i])

        # animation
        line_traj = [None] * self.num_uav
        c_x = [None] * self.num_uav
        c_y = [None] * self.num_uav

        r1_x = [None] * self.num_uav
        r1_y = [None] * self.num_uav

        r2_x = [None] * self.num_uav
        r2_y = [None] * self.num_uav

        line_arm1 = [None] * self.num_uav
        line_arm2 = [None] * self.num_uav

        line_traj_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            line_traj[i], = ax.plot(position[i][:1, 0], position[i][:1, 1])
            c_x[i], c_y[i] = position[i][0, 0:2]
            r1_x[i], r1_y[i] = position[i][0, 2:4]
            r2_x[i], r2_y[i] = position[i][0, 4:6]

            line_arm1[i], = ax.plot([c_x[i], r1_x[i]], [c_y[i], r1_y[i]], linewidth=1, color='red',
                                    marker='o', markersize=2)
            line_arm2[i], = ax.plot([c_x[i], r2_x[i]], [c_y[i], r2_y[i]], linewidth=1, color='blue',
                                    marker='o', markersize=2)

            # line_traj_ref[i], = ax.plot(position_ref[:1, 0], position_ref[:1, 1], position_ref[:1, 2], color='gray', alpha=0.5)
            # c_x_ref, c_y_ref, c_z_ref = position_ref[0, 0:3]
            # r1_x_ref, r1_y_ref, r1_z_ref = position_ref[0, 3:6]
            # r2_x_ref, r2_y_ref, r2_z_ref = position_ref[0, 6:9]
            # r3_x_ref, r3_y_ref, r3_z_ref = position_ref[0, 9:12]
            # r4_x_ref, r4_y_ref, r4_z_ref = position_ref[0, 12:15]
            # line_arm1_ref, = ax.plot([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref], [c_z_ref, r1_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm2_ref, = ax.plot([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref], [c_z_ref, r2_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm3_ref, = ax.plot([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref], [c_z_ref, r3_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm4_ref, = ax.plot([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref], [c_z_ref, r4_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)

        # time label
        time_template = 'time = %.1fs'
        time_text = ax.text(0.66, 0.55, "time", transform=ax.transAxes)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))
        sim_horizon = np.size(position[0], 0)

        def update_traj(num):
            # customize
            time_text.set_text(time_template % ((num+1) * 2 * dt))

            # trajectory
            for i in range(self.num_uav):
                line_traj[i].set_data(position[i][:num, 0], position[i][:num, 1])

                # uav
                c_x[i], c_y[i] = position[i][num, 0:2]
                r1_x[i], r1_y[i] = position[i][num, 2:4]
                r2_x[i], r2_y[i] = position[i][num, 4:6]

                line_arm1[i].set_data(np.array([[c_x[i], r1_x[i]], [c_y[i], r1_y[i]]]))

                line_arm2[i].set_data(np.array([[c_x[i], r2_x[i]], [c_y[i], r2_y[i]]]))

            # trajectory ref
            #     num = sim_horizon - 1
            #     line_traj_ref.set_data(position_ref[:num, 0], position_ref[:num, 1])
            #     line_traj_ref.set_3d_properties(position_ref[:num, 2])
            #
            #     # uav ref
            #     c_x_ref, c_y_ref, c_z_ref = position_ref[num, 0:3]
            #     r1_x_ref, r1_y_ref, r1_z_ref = position_ref[num, 3:6]
            #     r2_x_ref, r2_y_ref, r2_z_ref = position_ref[num, 6:9]
            #     r3_x_ref, r3_y_ref, r3_z_ref = position_ref[num, 9:12]
            #     r4_x_ref, r4_y_ref, r4_z_ref = position_ref[num, 12:15]
            #
            #     line_arm1_ref.set_data(np.array([[c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref]]))
            #     line_arm1_ref.set_3d_properties([c_z_ref, r1_z_ref])
            #
            #     line_arm2_ref.set_data(np.array([[c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref]]))
            #     line_arm2_ref.set_3d_properties([c_z_ref, r2_z_ref])
            #
            #     line_arm3_ref.set_data(np.array([[c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref]]))
            #     line_arm3_ref.set_3d_properties([c_z_ref, r3_z_ref])
            #
            #     line_arm4_ref.set_data(np.array([[c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref]]))
            #     line_arm4_ref.set_3d_properties([c_z_ref, r4_z_ref])

            # if num == sim_horizon-2:
            #    plt.pause(2)

            return line_traj[0], line_arm1[0], line_arm2[0], line_traj[1], line_arm1[1], \
                   line_arm2[1], line_traj[2], line_arm1[2], line_arm2[2], line_traj[3], line_arm1[3], line_arm2[
                       3], time_text  # , \

            # line_traj_ref, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=200, blit=True)

        def restart():
            root = tkinter.Tk()
            root.withdraw()
            result = tkinter.messagebox.askyesno("Restart", "Do you want to restart animation?")
            if result:
                ani.frame_seq = ani.new_frame_seq()
                ani.event_source.start()
            else:
                plt.close()

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('case2' + title + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()

    def dir_cosine(self, q):
        C_B_I = vertcat(
            horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
        return C_B_I

    def skew(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2], v[1]),
            horzcat(v[2], 0, -v[0]),
            horzcat(-v[1], v[0], 0)
        )
        return v_cross

    def omega(self, w):
        omeg = vertcat(
            horzcat(0, -w[0], -w[1], -w[2]),
            horzcat(w[0], 0, w[2], -w[1]),
            horzcat(w[1], -w[2], 0, w[0]),
            horzcat(w[2], w[1], -w[0], 0)
        )
        return omeg

    def quaternion_mul(self, p, q):
        return vertcat(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                       p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                       p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                       p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
                       )

class UAV_formation_transition_diamond_2D:
    def __init__(self, project_name='my UAV swarm 2D'):
        self.project_name = 'my uav swarm 2D'

        self.states_bd = [None, None]
        self.ctrl_bd = [None, None]
        self.UAV_labels = None
        self.obstacles = [[4.3, 3], [4.8, 3]]
        self.wall = [0, 5]

    def initDyn(self, uav_index=0, Jxy=None, mass=None, l=0.1, states_bd=[None, None],
                ctrl_bd=[None, None], dt=1e-3):
        # define the state of the uavs
        # Model reference : http://underactuated.mit.edu/acrobot.html
        rx, ry = SX.sym('rx' + str(uav_index)), SX.sym('ry' + str(uav_index))
        self.r_I = vertcat(rx, ry)
        vx, vy = SX.sym('vx' + str(uav_index)), SX.sym('vy' + str(uav_index))
        self.v_I = vertcat(vx, vy)

        self.theta = SX.sym('theta' + str(uav_index))
        self.w_B = SX.sym('w' + str(uav_index))
        # define the Planar quadrotor input
        f1, f2 = SX.sym('f1' + str(uav_index)), SX.sym('f2' + str(uav_index))
        self.T_B = vertcat(f1, f2)

        # global parameter
        g = 10

        # parameters settings
        parameter = []
        if Jxy is None:
            self.Jxy = SX.sym('Jxy' + str(uav_index))
            parameter += [self.Jxy]
        else:
            self.Jxy = Jxy

        if mass is None:
            self.mass = SX.sym('mass' + str(uav_index))
            parameter += [self.mass]
        else:
            self.mass = mass

        if l is None:
            self.l = SX.sym('l' + str(uav_index))
            parameter += [self.l]
        else:
            self.l = l

        self.dyn_auxvar = vcat(parameter)

        # Angular moment of inertia
        # self.J_B = diag(vertcat(self.Jx, self.Jy))
        # Gravity
        self.g_I = vertcat(0, -g)
        # Mass of rocket, assume is little changed during the landing process
        self.m = self.mass

        dr_I = self.v_I
        dv_I = 1 / self.m * vertcat(-sin(self.theta) * (self.T_B[0] + self.T_B[1]),
                                    cos(self.theta) * (self.T_B[0] + self.T_B[1])) + self.g_I
        dtheta = self.w_B
        dw_B = 1 / self.Jxy * self.l * (self.T_B[0] - self.T_B[1])

        self.states = vertcat(self.r_I, self.v_I, self.theta, self.w_B)

        if np.asarray(states_bd).size == self.states.numel():
            self.states_bd = states_bd
        else:
            self.states_bd = self.states.numel() * [-1e10, 1e10]

        self.ctrl = self.T_B
        if np.asarray(ctrl_bd).size == self.ctrl.numel():
            self.ctrl_bd = ctrl_bd
        else:
            self.states_bd = self.states.numel() * [-1e10, 1e10]

        self.dynf = vertcat(dr_I, dv_I, dtheta, dw_B)
        self.dt = dt

    def initCost(self, adj_matrix=None, uavswarm=None, num_uav=None, uav_index=None, w_F_r=None, w_F_v=None,
                 w_r_formation=None,
                 w_v_formation=None,
                 uav_dist=None, w_uav_collision=None, w_obst=None, wthrust=0.1):

        parameter = []
        if w_F_r is None:
            self.w_F_r = SX.sym('w_F_r' + str(uav_index))
            parameter += [self.w_F_r]
        else:
            self.w_F_r = w_F_r

        if w_F_v is None:
            self.w_F_v = SX.sym('w_F_v' + str(uav_index))
            parameter += [self.w_F_v]
        else:
            self.w_F_v = w_F_v

        if w_r_formation is None:
            self.w_r_formation = SX.sym('w_r_formation' + str(uav_index))
            parameter += [self.w_r_formation]
        else:
            self.w_r_formation = w_r_formation

        if w_v_formation is None:
            self.w_v_formation = SX.sym('w_v_formation' + str(uav_index))
            parameter += [self.w_v_formation]
        else:
            self.w_v_formation = w_v_formation

        if uav_dist is None:
            self.uav_dist = SX.sym('uav_dist' + str(uav_index))
            parameter += [self.uav_dist]
        else:
            self.uav_dist = uav_dist

        if w_uav_collision is None:
            self.w_uav_collision = SX.sym('w_uav_collision' + str(uav_index))
            parameter += [self.w_uav_collision]
        else:
            self.w_uav_collision = w_uav_collision

        if w_obst is None:
            self.w_obst = SX.sym('w_obst' + str(uav_index))
            parameter += [self.w_obst]
        else:
            self.w_obst = w_obst

        self.cost_auxvar = vcat(parameter)

        # neighbors

        # formation distance and speed costs
        if adj_matrix:
            neighbors = list(np.where(np.array(adj_matrix) == 1)[0])
        else:
            neighbors = [*range(num_uav)]
        neighbors.remove(uav_index)

        self.cost_r_formation = 0
        self.cost_v_formation = 0
        self.cost_v_formation += dot(self.v_I[1] - 2, self.v_I[1] - 2)

        if uav_index == 1:
            self.cost_r_formation += dot(self.r_I[1] - uavswarm[3].r_I[1],self.r_I[1] - uavswarm[3].r_I[1])
            self.cost_r_formation += dot(dot(self.r_I - uavswarm[2].r_I, self.r_I - uavswarm[2].r_I) - self.uav_dist**2,
                                         dot(self.r_I - uavswarm[2].r_I, self.r_I - uavswarm[2].r_I) - self.uav_dist**2)
            self.cost_r_formation += dot(dot(self.r_I - uavswarm[3].r_I, self.r_I - uavswarm[3].r_I) - self.uav_dist**3,
                                         dot(self.r_I - uavswarm[3].r_I, self.r_I - uavswarm[3].r_I) - self.uav_dist**3)
        elif uav_index == 2:
            self.cost_r_formation += dot(self.r_I - uavswarm[3].r_I - vcat([-0.866 * self.uav_dist, 0.5 * self.uav_dist]),
                                         self.r_I - uavswarm[3].r_I - vcat([-0.866 * self.uav_dist, 0.5 * self.uav_dist]))
        elif uav_index == 3:
            self.cost_r_formation += dot(self.r_I - uavswarm[2].r_I - vcat([0.866 * self.uav_dist, -0.5 * self.uav_dist]),
                                         self.r_I - uavswarm[2].r_I - vcat([0.866 * self.uav_dist, -0.5 * self.uav_dist]))
        elif uav_index == 0:
            self.cost_r_formation += dot((self.r_I + uavswarm[2].r_I)/2 - uavswarm[1].r_I,
                                         (self.r_I + uavswarm[2].r_I)/2 - uavswarm[1].r_I)


        for j in neighbors:
            self.cost_v_formation += dot(self.v_I - uavswarm[j].v_I, self.v_I - uavswarm[j].v_I)


        goal_r1_I = np.array([2.8, 8])
        goal_r2_I = np.array([2.8+0.8*1.732, 8-0.8])

        if uav_index == 0:
            self.cost_r_I = dot((self.r_I + goal_r1_I) / 2 - uavswarm[1].r_I,
                                (self.r_I + goal_r1_I) / 2 - uavswarm[1].r_I)
        elif uav_index == 1:
            self.cost_r_I = dot(dot(self.r_I - uavswarm[2].r_I, self.r_I - uavswarm[2].r_I) - self.uav_dist**2,
                                dot(self.r_I - uavswarm[2].r_I, self.r_I - uavswarm[2].r_I) - self.uav_dist**2)
            self.cost_r_I += dot(dot(self.r_I - uavswarm[3].r_I, self.r_I - uavswarm[3].r_I) - self.uav_dist ** 3,
                                dot(self.r_I - uavswarm[3].r_I, self.r_I - uavswarm[3].r_I) - self.uav_dist ** 3)
        elif uav_index == 2:
            self.cost_r_I = dot(self.r_I - goal_r1_I, self.r_I - goal_r1_I)

        elif uav_index == 3:
            self.cost_r_I = dot(self.r_I - goal_r2_I, self.r_I - goal_r2_I)




        # else:
        #    self.cost_r_I =0

        #    self.cost_r_I = 0
        # goal velocity cost
        goal_v_I = np.array([0, 2])
        self.cost_v_I = dot(self.v_I - goal_v_I, self.v_I - goal_v_I)

        # collision cost
        self.cost_collision = 0
        for i in neighbors:
            # self.cost_collision = self.cost_collision + 1 / fmin(fmax(norm_2(self.r_I - uavswarm[i].r_I), 1e-3),
            #                                                     0.6) - 1 / 0.6
            self.cost_collision = self.cost_collision + 1 / dot(self.r_I - uavswarm[i].r_I, self.r_I - uavswarm[i].r_I)

        self.cost_obst = 1 / dot(self.r_I[0:2] - self.obstacles[0], self.r_I[0:2] - self.obstacles[0]) + \
                         1 / dot(self.r_I[0:2] - self.obstacles[1], self.r_I[0:2] - self.obstacles[1])
        # the thrust cost
        self.cost_thrust = dot(self.T_B, self.T_B)

        self.path_cost = self.w_r_formation * self.cost_r_formation + \
                         self.w_v_formation * self.cost_v_formation + \
                         self.w_uav_collision * self.cost_collision + \
                         self.w_obst * self.cost_obst + \
                         wthrust * self.cost_thrust
        # self.w_v_formation * self.cost_v_formation + \
        # self.w_F_r * self.cost_r_I + \
        # self.w_F_v * self.cost_v_I + \

        self.final_cost = self.w_F_r * self.cost_r_I + \
                          self.w_F_v * self.cost_v_I + \
                          10 * self.w_r_formation * self.cost_r_formation + \
                          10 * self.w_v_formation * self.cost_v_formation + \
                          self.w_uav_collision * self.cost_collision

    def get_UAV_position(self, wing_len, state_traj):

        # thrust_position in body frame

        # horizon
        horizon = np.size(state_traj, 0)
        position = np.zeros((horizon, 6))
        for t in range(horizon):
            # position of COM
            rc = state_traj[t, 0:2]
            # altitude of quaternion
            theta = state_traj[t, 4]

            # position of each rotor in inertial frame
            r1_pos = rc + vertcat(-cos(theta) * wing_len / 2, -sin(theta) * wing_len / 2).T
            r2_pos = rc + vertcat(cos(theta) * wing_len / 2, sin(theta) * wing_len / 2).T

            # store
            position[t, 0:2] = rc
            position[t, 2:4] = r1_pos
            position[t, 4:6] = r2_pos

        return position

    def play_multi_animation(self, uavswarm, wing_len, state_traj, state_traj_ref=None, dt=None, save_option=0,
                             title='UAV Maneuvering'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_ylim(-1, 10)
        ax.set_xlim(-1, 6)
        ax.set_box_aspect(1)
        ax.set_title(title, pad=20, fontsize=15)

        wall1 = plt.Rectangle((0, -1), 0.1, 11, color='limegreen', alpha=0.2)
        wall2 = plt.Rectangle((5, -1), 0.1, 11, color='limegreen', alpha=0.2)
        risky = plt.Rectangle((self.obstacles[0][0] - 0.5, self.obstacles[0][1] - 0.8), 1.5, 1, color='r', alpha=0.5)

        ax.add_patch(wall1)
        ax.add_patch(wall2)
        ax.add_patch(risky)

        # data
        self.num_uav = len(uavswarm)
        position = [None] * self.num_uav
        position_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            position[i] = self.get_UAV_position(wing_len, state_traj[i])
            if state_traj_ref is None:
                position_ref[i] = self.get_UAV_position(0, numpy.zeros_like(position[i]))
            else:
                position_ref[i] = self.get_UAV_position(wing_len, state_traj_ref[i])

        # animation
        line_traj = [None] * self.num_uav
        c_x = [None] * self.num_uav
        c_y = [None] * self.num_uav

        r1_x = [None] * self.num_uav
        r1_y = [None] * self.num_uav

        r2_x = [None] * self.num_uav
        r2_y = [None] * self.num_uav

        line_arm1 = [None] * self.num_uav
        line_arm2 = [None] * self.num_uav

        line_traj_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            line_traj[i], = ax.plot(position[i][:1, 0], position[i][:1, 1])
            c_x[i], c_y[i] = position[i][0, 0:2]
            r1_x[i], r1_y[i] = position[i][0, 2:4]
            r2_x[i], r2_y[i] = position[i][0, 4:6]

            line_arm1[i], = ax.plot([c_x[i], r1_x[i]], [c_y[i], r1_y[i]], linewidth=1, color='red',
                                    marker='o', markersize=2)
            line_arm2[i], = ax.plot([c_x[i], r2_x[i]], [c_y[i], r2_y[i]], linewidth=1, color='blue',
                                    marker='o', markersize=2)

            # line_traj_ref[i], = ax.plot(position_ref[:1, 0], position_ref[:1, 1], position_ref[:1, 2], color='gray', alpha=0.5)
            # c_x_ref, c_y_ref, c_z_ref = position_ref[0, 0:3]
            # r1_x_ref, r1_y_ref, r1_z_ref = position_ref[0, 3:6]
            # r2_x_ref, r2_y_ref, r2_z_ref = position_ref[0, 6:9]
            # r3_x_ref, r3_y_ref, r3_z_ref = position_ref[0, 9:12]
            # r4_x_ref, r4_y_ref, r4_z_ref = position_ref[0, 12:15]
            # line_arm1_ref, = ax.plot([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref], [c_z_ref, r1_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm2_ref, = ax.plot([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref], [c_z_ref, r2_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm3_ref, = ax.plot([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref], [c_z_ref, r3_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm4_ref, = ax.plot([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref], [c_z_ref, r4_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)

        # time label
        time_template = 'time = %.1fs'
        time_text = ax.text(0.66, 0.55, "time", transform=ax.transAxes)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))
        sim_horizon = np.size(position[0], 0)

        def update_traj(num):
            # customize
            time_text.set_text(time_template % (num * dt))

            # trajectory
            for i in range(self.num_uav):
                line_traj[i].set_data(position[i][:num, 0], position[i][:num, 1])

                # uav
                c_x[i], c_y[i] = position[i][num, 0:2]
                r1_x[i], r1_y[i] = position[i][num, 2:4]
                r2_x[i], r2_y[i] = position[i][num, 4:6]

                line_arm1[i].set_data(np.array([[c_x[i], r1_x[i]], [c_y[i], r1_y[i]]]))

                line_arm2[i].set_data(np.array([[c_x[i], r2_x[i]], [c_y[i], r2_y[i]]]))

            # trajectory ref
            #     num = sim_horizon - 1
            #     line_traj_ref.set_data(position_ref[:num, 0], position_ref[:num, 1])
            #     line_traj_ref.set_3d_properties(position_ref[:num, 2])
            #
            #     # uav ref
            #     c_x_ref, c_y_ref, c_z_ref = position_ref[num, 0:3]
            #     r1_x_ref, r1_y_ref, r1_z_ref = position_ref[num, 3:6]
            #     r2_x_ref, r2_y_ref, r2_z_ref = position_ref[num, 6:9]
            #     r3_x_ref, r3_y_ref, r3_z_ref = position_ref[num, 9:12]
            #     r4_x_ref, r4_y_ref, r4_z_ref = position_ref[num, 12:15]
            #
            #     line_arm1_ref.set_data(np.array([[c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref]]))
            #     line_arm1_ref.set_3d_properties([c_z_ref, r1_z_ref])
            #
            #     line_arm2_ref.set_data(np.array([[c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref]]))
            #     line_arm2_ref.set_3d_properties([c_z_ref, r2_z_ref])
            #
            #     line_arm3_ref.set_data(np.array([[c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref]]))
            #     line_arm3_ref.set_3d_properties([c_z_ref, r3_z_ref])
            #
            #     line_arm4_ref.set_data(np.array([[c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref]]))
            #     line_arm4_ref.set_3d_properties([c_z_ref, r4_z_ref])

            # if num == sim_horizon-2:
            #    plt.pause(2)

            return line_traj[0], line_arm1[0], line_arm2[0], line_traj[1], line_arm1[1], \
                   line_arm2[1], line_traj[2], line_arm1[2], line_arm2[2], line_traj[3], line_arm1[3], line_arm2[
                       3], time_text  # , \

            # line_traj_ref, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=100, blit=True)

        def restart():
            root = tkinter.Tk()
            root.withdraw()
            result = tkinter.messagebox.askyesno("Restart", "Do you want to restart animation?")
            if result:
                ani.frame_seq = ani.new_frame_seq()
                ani.event_source.start()
            else:
                plt.close()

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('case2' + title + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()

    def play_multi_animation_3(self, uavswarm, wing_len, state_traj, state_traj_ref=None, dt=None, save_option=0,
                               title='UAV Maneuvering'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_ylim(-1, 10)
        ax.set_xlim(-1, 6)
        ax.set_box_aspect(1)
        ax.set_title(title, pad=20, fontsize=15)

        wall1 = plt.Rectangle((0, -1), 0.1, 11, color='limegreen', alpha=0.2)
        wall2 = plt.Rectangle((5, -1), 0.1, 11, color='limegreen', alpha=0.2)
        risky = plt.Rectangle((self.obstacles[0][0] - 0.5, self.obstacles[0][1] - 0.8), 1.5, 1, color='r', alpha=0.5)

        ax.add_patch(wall1)
        ax.add_patch(wall2)
        ax.add_patch(risky)

        # data
        self.num_uav = len(uavswarm)
        position = [None] * self.num_uav
        position_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            position[i] = self.get_UAV_position(wing_len, state_traj[i])
            if state_traj_ref is None:
                position_ref[i] = self.get_UAV_position(0, numpy.zeros_like(position[i]))
            else:
                position_ref[i] = self.get_UAV_position(wing_len, state_traj_ref[i])

        # animation
        line_traj = [None] * self.num_uav
        c_x = [None] * self.num_uav
        c_y = [None] * self.num_uav

        r1_x = [None] * self.num_uav
        r1_y = [None] * self.num_uav

        r2_x = [None] * self.num_uav
        r2_y = [None] * self.num_uav

        line_arm1 = [None] * self.num_uav
        line_arm2 = [None] * self.num_uav

        line_traj_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            line_traj[i], = ax.plot(position[i][:1, 0], position[i][:1, 1])
            c_x[i], c_y[i] = position[i][0, 0:2]
            r1_x[i], r1_y[i] = position[i][0, 2:4]
            r2_x[i], r2_y[i] = position[i][0, 4:6]

            line_arm1[i], = ax.plot([c_x[i], r1_x[i]], [c_y[i], r1_y[i]], linewidth=1, color='red',
                                    marker='o', markersize=2)
            line_arm2[i], = ax.plot([c_x[i], r2_x[i]], [c_y[i], r2_y[i]], linewidth=1, color='blue',
                                    marker='o', markersize=2)

            # line_traj_ref[i], = ax.plot(position_ref[:1, 0], position_ref[:1, 1], position_ref[:1, 2], color='gray', alpha=0.5)
            # c_x_ref, c_y_ref, c_z_ref = position_ref[0, 0:3]
            # r1_x_ref, r1_y_ref, r1_z_ref = position_ref[0, 3:6]
            # r2_x_ref, r2_y_ref, r2_z_ref = position_ref[0, 6:9]
            # r3_x_ref, r3_y_ref, r3_z_ref = position_ref[0, 9:12]
            # r4_x_ref, r4_y_ref, r4_z_ref = position_ref[0, 12:15]
            # line_arm1_ref, = ax.plot([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref], [c_z_ref, r1_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm2_ref, = ax.plot([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref], [c_z_ref, r2_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm3_ref, = ax.plot([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref], [c_z_ref, r3_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm4_ref, = ax.plot([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref], [c_z_ref, r4_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)

        # time label
        time_template = 'time = %.1fs'
        time_text = ax.text(0.66, 0.55, "time", transform=ax.transAxes)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))
        sim_horizon = np.size(position[0], 0)

        def update_traj(num):
            # customize
            time_text.set_text(time_template % (num * dt))

            # trajectory
            for i in range(self.num_uav):
                line_traj[i].set_data(position[i][:num, 0], position[i][:num, 1])

                # uav
                c_x[i], c_y[i] = position[i][num, 0:2]
                r1_x[i], r1_y[i] = position[i][num, 2:4]
                r2_x[i], r2_y[i] = position[i][num, 4:6]

                line_arm1[i].set_data(np.array([[c_x[i], r1_x[i]], [c_y[i], r1_y[i]]]))

                line_arm2[i].set_data(np.array([[c_x[i], r2_x[i]], [c_y[i], r2_y[i]]]))

            # trajectory ref
            #     num = sim_horizon - 1
            #     line_traj_ref.set_data(position_ref[:num, 0], position_ref[:num, 1])
            #     line_traj_ref.set_3d_properties(position_ref[:num, 2])
            #
            #     # uav ref
            #     c_x_ref, c_y_ref, c_z_ref = position_ref[num, 0:3]
            #     r1_x_ref, r1_y_ref, r1_z_ref = position_ref[num, 3:6]
            #     r2_x_ref, r2_y_ref, r2_z_ref = position_ref[num, 6:9]
            #     r3_x_ref, r3_y_ref, r3_z_ref = position_ref[num, 9:12]
            #     r4_x_ref, r4_y_ref, r4_z_ref = position_ref[num, 12:15]
            #
            #     line_arm1_ref.set_data(np.array([[c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref]]))
            #     line_arm1_ref.set_3d_properties([c_z_ref, r1_z_ref])
            #
            #     line_arm2_ref.set_data(np.array([[c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref]]))
            #     line_arm2_ref.set_3d_properties([c_z_ref, r2_z_ref])
            #
            #     line_arm3_ref.set_data(np.array([[c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref]]))
            #     line_arm3_ref.set_3d_properties([c_z_ref, r3_z_ref])
            #
            #     line_arm4_ref.set_data(np.array([[c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref]]))
            #     line_arm4_ref.set_3d_properties([c_z_ref, r4_z_ref])

            # if num == sim_horizon-2:
            #    plt.pause(2)

            return line_traj[0], line_arm1[0], line_arm2[0], line_traj[1], line_arm1[1], \
                   line_arm2[1], line_traj[2], line_arm1[2], line_arm2[2], line_traj[3], line_arm1[3], line_arm2[
                       3], time_text  # , \

            # line_traj_ref, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=100, blit=True)

        def restart():
            root = tkinter.Tk()
            root.withdraw()
            result = tkinter.messagebox.askyesno("Restart", "Do you want to restart animation?")
            if result:
                ani.frame_seq = ani.new_frame_seq()
                ani.event_source.start()
            else:
                plt.close()

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('case2' + title + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()

    def play_multi_animation_tran(self, uavswarm, wing_len, state_traj, state_traj_ref=None, dt=None, save_option=0,
                               title='UAV Maneuvering'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_ylim(-1, 10)
        ax.set_xlim(-1, 6)
        ax.set_box_aspect(1)
        ax.set_title(title, pad=20, fontsize=15)

        wall1 = plt.Rectangle((0, -1), 0.1, 11, color='limegreen', alpha=0.2)
        wall2 = plt.Rectangle((5, -1), 0.1, 11, color='limegreen', alpha=0.2)
        risky = plt.Rectangle((self.obstacles[0][0] - 0.25, self.obstacles[0][1] - 0.3), 1.05, 0.6, color='r', alpha=0.5)

        ax.add_patch(wall1)
        ax.add_patch(wall2)
        ax.add_patch(risky)

        # data
        self.num_uav = len(uavswarm)
        position = [None] * self.num_uav
        position_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            position[i] = self.get_UAV_position(wing_len, state_traj[i])
            if state_traj_ref is None:
                position_ref[i] = self.get_UAV_position(0, numpy.zeros_like(position[i]))
            else:
                position_ref[i] = self.get_UAV_position(wing_len, state_traj_ref[i])

        # animation
        line_traj = [None] * self.num_uav
        c_x = [None] * self.num_uav
        c_y = [None] * self.num_uav

        r1_x = [None] * self.num_uav
        r1_y = [None] * self.num_uav

        r2_x = [None] * self.num_uav
        r2_y = [None] * self.num_uav

        line_arm1 = [None] * self.num_uav
        line_arm2 = [None] * self.num_uav

        line_traj_ref = [None] * self.num_uav
        for i in range(self.num_uav):
            line_traj[i], = ax.plot(position[i][:1, 0], position[i][:1, 1])
            c_x[i], c_y[i] = position[i][0, 0:2]
            r1_x[i], r1_y[i] = position[i][0, 2:4]
            r2_x[i], r2_y[i] = position[i][0, 4:6]

            line_arm1[i], = ax.plot([c_x[i], r1_x[i]], [c_y[i], r1_y[i]], linewidth=1, color='red',
                                    marker='o', markersize=2)
            line_arm2[i], = ax.plot([c_x[i], r2_x[i]], [c_y[i], r2_y[i]], linewidth=1, color='blue',
                                    marker='o', markersize=2)

            # line_traj_ref[i], = ax.plot(position_ref[:1, 0], position_ref[:1, 1], position_ref[:1, 2], color='gray', alpha=0.5)
            # c_x_ref, c_y_ref, c_z_ref = position_ref[0, 0:3]
            # r1_x_ref, r1_y_ref, r1_z_ref = position_ref[0, 3:6]
            # r2_x_ref, r2_y_ref, r2_z_ref = position_ref[0, 6:9]
            # r3_x_ref, r3_y_ref, r3_z_ref = position_ref[0, 9:12]
            # r4_x_ref, r4_y_ref, r4_z_ref = position_ref[0, 12:15]
            # line_arm1_ref, = ax.plot([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref], [c_z_ref, r1_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm2_ref, = ax.plot([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref], [c_z_ref, r2_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm3_ref, = ax.plot([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref], [c_z_ref, r3_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)
            # line_arm4_ref, = ax.plot([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref], [c_z_ref, r4_z_ref], linewidth=2,
            #                      color='gray', marker='o', markersize=3, alpha=0.7)

        # time label
        time_template = 'time = %.1fs'
        time_text = ax.text(0.66, 0.55, "time", transform=ax.transAxes)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))
        sim_horizon = np.size(position[0], 0)

        def update_traj(num):
            # customize
            time_text.set_text(time_template % ((num+1)*2 * dt))

            # trajectory
            for i in range(self.num_uav):
                line_traj[i].set_data(position[i][:num, 0], position[i][:num, 1])

                # uav
                c_x[i], c_y[i] = position[i][num, 0:2]
                r1_x[i], r1_y[i] = position[i][num, 2:4]
                r2_x[i], r2_y[i] = position[i][num, 4:6]

                line_arm1[i].set_data(np.array([[c_x[i], r1_x[i]], [c_y[i], r1_y[i]]]))

                line_arm2[i].set_data(np.array([[c_x[i], r2_x[i]], [c_y[i], r2_y[i]]]))

            # trajectory ref
            #     num = sim_horizon - 1
            #     line_traj_ref.set_data(position_ref[:num, 0], position_ref[:num, 1])
            #     line_traj_ref.set_3d_properties(position_ref[:num, 2])
            #
            #     # uav ref
            #     c_x_ref, c_y_ref, c_z_ref = position_ref[num, 0:3]
            #     r1_x_ref, r1_y_ref, r1_z_ref = position_ref[num, 3:6]
            #     r2_x_ref, r2_y_ref, r2_z_ref = position_ref[num, 6:9]
            #     r3_x_ref, r3_y_ref, r3_z_ref = position_ref[num, 9:12]
            #     r4_x_ref, r4_y_ref, r4_z_ref = position_ref[num, 12:15]
            #
            #     line_arm1_ref.set_data(np.array([[c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref]]))
            #     line_arm1_ref.set_3d_properties([c_z_ref, r1_z_ref])
            #
            #     line_arm2_ref.set_data(np.array([[c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref]]))
            #     line_arm2_ref.set_3d_properties([c_z_ref, r2_z_ref])
            #
            #     line_arm3_ref.set_data(np.array([[c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref]]))
            #     line_arm3_ref.set_3d_properties([c_z_ref, r3_z_ref])
            #
            #     line_arm4_ref.set_data(np.array([[c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref]]))
            #     line_arm4_ref.set_3d_properties([c_z_ref, r4_z_ref])

            # if num == sim_horizon-2:
            #    plt.pause(2)

            return line_traj[0], line_arm1[0], line_arm2[0], line_traj[1], line_arm1[1], \
                   line_arm2[1], line_traj[2], line_arm1[2], line_arm2[2], line_traj[3], line_arm1[3], line_arm2[
                       3], time_text  # , \

            # line_traj_ref, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=200, blit=True)

        def restart():
            root = tkinter.Tk()
            root.withdraw()
            result = tkinter.messagebox.askyesno("Restart", "Do you want to restart animation?")
            if result:
                ani.frame_seq = ani.new_frame_seq()
                ani.event_source.start()
            else:
                plt.close()

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('case2' + title + '.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()

    def dir_cosine(self, q):
        C_B_I = vertcat(
            horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
        return C_B_I

    def skew(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2], v[1]),
            horzcat(v[2], 0, -v[0]),
            horzcat(-v[1], v[0], 0)
        )
        return v_cross

    def omega(self, w):
        omeg = vertcat(
            horzcat(0, -w[0], -w[1], -w[2]),
            horzcat(w[0], 0, w[2], -w[1]),
            horzcat(w[1], -w[2], 0, w[0]),
            horzcat(w[2], w[1], -w[0], 0)
        )
        return omeg

    def quaternion_mul(self, p, q):
        return vertcat(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                       p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                       p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                       p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
                       )



# converter to quaternion from (angle, direction)
def toQuaternion(angle, dir):
    if type(dir) == list:
        dir = numpy.array(dir)
    dir = dir / numpy.linalg.norm(dir)
    quat = numpy.zeros(4)
    quat[0] = math.cos(angle / 2)
    quat[1:] = math.sin(angle / 2) * dir
    return quat.tolist()


# normalized verctor
def normalizeVec(vec):
    if type(vec) == list:
        vec = np.array(vec)
    vec = vec / np.linalg.norm(vec)
    return vec


def quaternion_conj(q):
    conj_q = q
    conj_q[1] = -q[1]
    conj_q[2] = -q[2]
    conj_q[3] = -q[3]
    return conj_q
