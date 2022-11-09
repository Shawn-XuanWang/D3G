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


class UAV_formation():

    def __init__(self, uav_num, project_name='my UAV swarm'):
        self.project_name = 'my uav swarm'
        self.uav_num = uav_num
        self.states_bd = [None, None]
        self.ctrl_bd = [None, None]
        self.UAV_labels = None
        self.r_I = [None] * self.uav_num
        self.v_I = [None] * self.uav_num
        self.w_B = [None] * self.uav_num
        self.q = [None] * self.uav_num
        self.states = [None] * self.uav_num


        #obstacles
        self.obstacles = [[4.3, 3], [4.8, 3]]
        self.wall = [0, 5]


    def setRandomObstacles(self, n_obstacle=3, ball_r=1.0):

        self.obstacle_plot_info = []
        self.obstacle_info = []
        for i in range(n_obstacle):
            # ball position
            ball_xyz = np.random.uniform(low=np.array([-5.0, -5.0, 0]) + ball_r,
                                         high=np.array([5.0, 5.0, 15]) - ball_r,
                                         size=(3,))
            self.obstacle_info.append([ball_xyz, ball_r])

            # ball surface
            self.obstacle_plot_info.append(self._3DBall_surface(ball_xyz, ball_r))

    # allow you set the random obstacles
    def setObstacles(self, obstacle_xyz, ball_r):

        self.obstacle_plot_info = []
        self.obstacle_info = []

        n_obstacle = len(obstacle_xyz)
        for i in range(n_obstacle):
            ball_xyz = obstacle_xyz[i]
            self.obstacle_info.append([ball_xyz, ball_r])

            # ball surface
            self.obstacle_plot_info.append(self._3DBall_surface(ball_xyz, ball_r))

    def initDyn(self, Jx=None, Jy=None, Jz=None, mass=None, l=0.1, c=None, states_bd=[None, None],
                ctrl_bd=[None, None], dt=1e-3):

        #define the parameter
        self.Jx = [None] * self.uav_num
        self.Jy = [None] * self.uav_num
        self.Jz = [None] * self.uav_num
        self.mass = [None] * self.uav_num
        self.l = [None] * self.uav_num
        self.c = [None] * self.uav_num
        self.dyn_auxvar = [None] * self.uav_num
        self.J_B = [None] * self.uav_num
        self.m = [None] * self.uav_num
        self.thrust_B = [None] * self.uav_num
        self.M_B = [None] * self.uav_num
        self.T_B = [None] * self.uav_num
        self.dynf = [None] * self.uav_num
        self.ctrl = [None] * self.uav_num
        self.states_bd = [None] * self.uav_num
        self.ctrl_bd = [None] * self.uav_num

        # global parameter
        g = 10
        self.dt = dt
        # Gravity
        self.g_I = vertcat(0, 0, -g)

        #set parameter for each uav
        for uav_index in range(self.uav_num):
            parameter = []

            rx, ry, rz = SX.sym('rx' + str(uav_index)), SX.sym('ry' + str(uav_index)), SX.sym('rz' + str(uav_index))
            self.r_I[uav_index] = vertcat(rx, ry, rz)
            vx, vy, vz = SX.sym('vx' + str(uav_index)), SX.sym('vy' + str(uav_index)), SX.sym('vz' + str(uav_index))
            self.v_I[uav_index] = vertcat(vx, vy, vz)

            # quaternions attitude of B w.r.t. I
            q0, q1, q2, q3 = SX.sym('q0' + str(uav_index)), SX.sym('q1' + str(uav_index)), SX.sym(
                'q2' + str(uav_index)), SX.sym('q3' + str(uav_index))
            self.q[uav_index] = vertcat(q0, q1, q2, q3)
            wx, wy, wz = SX.sym('wx' + str(uav_index)), SX.sym('wy' + str(uav_index)), SX.sym('wz' + str(uav_index))
            self.w_B[uav_index] = vertcat(wx, wy, wz)

            # define the quadrotor input
            f1, f2, f3, f4 = SX.sym('f1' + str(uav_index)), SX.sym('f2' + str(uav_index)), SX.sym(
                'f3' + str(uav_index)), SX.sym('f4' + str(uav_index))
            self.T_B[uav_index] = vertcat(f1, f2, f3, f4)

            # parameters settings
            if Jx is None:
                self.Jx[uav_index] = SX.sym('Jx' + str(uav_index))
                parameter += [self.Jx[uav_index]]
            else:
                self.Jx[uav_index] = Jx

            if Jy is None:
                self.Jy[uav_index] = SX.sym('Jy' + str(uav_index))
                parameter += [self.Jy[uav_index]]
            else:
                self.Jy[uav_index] = Jy

            if Jz is None:
                self.Jz[uav_index] = SX.sym('Jz' + str(uav_index))
                parameter += [self.Jz[uav_index]]
            else:
                self.Jz[uav_index] = Jz

            if mass is None:
                self.mass[uav_index] = SX.sym('mass' + str(uav_index))
                parameter += [self.mass[uav_index]]
            else:
                self.mass[uav_index] = mass

            if l is None:
                self.l[uav_index] = SX.sym('l' + str(uav_index))
                parameter += [self.l[uav_index]]
            else:
                self.l[uav_index] = l

            if c is None:
                self.c[uav_index] = SX.sym('c' + str(uav_index))
                parameter += [self.c[uav_index]]
            else:
                self.c[uav_index] = c

            self.dyn_auxvar[uav_index] = vcat(parameter) # parameter:[Jx Jy Jz mass l c]

            # Angular moment of inertia
            self.J_B[uav_index] = diag(vertcat(self.Jx[uav_index], self.Jy[uav_index], self.Jz[uav_index]))

            # Mass of rocket, assume is little changed during the landing process
            self.m[uav_index] = self.mass[uav_index]

            # total thrust in body frame
            thrust = self.T_B[uav_index][0] + self.T_B[uav_index][1] + self.T_B[uav_index][2] + self.T_B[uav_index][3]
            self.thrust_B[uav_index] = vertcat(0, 0, thrust)
            # print(self.thrust_B[uav_index])

            # total moment M in body frame
            Mx = -self.T_B[uav_index][1] * self.l[uav_index] / 2 + self.T_B[uav_index][3] * self.l[uav_index] / 2
            My = -self.T_B[uav_index][0] * self.l[uav_index] / 2 + self.T_B[uav_index][2] * self.l[uav_index] / 2
            Mz = (self.T_B[uav_index][0] - self.T_B[uav_index][1] + self.T_B[uav_index][2] - self.T_B[uav_index][3]) * self.c[uav_index]
            self.M_B[uav_index] = vertcat(Mx, My, Mz)

            # cosine directional matrix
            C_B_I = self.dir_cosine(self.q[uav_index])  # inertial to body
            C_I_B = transpose(C_B_I)  # body to inertial

            # Newton's law
            dr_I = self.v_I[uav_index]
            dv_I = 1 / self.m[uav_index] * mtimes(C_I_B, self.thrust_B[uav_index]) + self.g_I

            # Euler's law
            dq = 1 / 2 * mtimes(self.omega(self.w_B[uav_index]), self.q[uav_index])
            dw = mtimes(inv(self.J_B[uav_index]), self.M_B[uav_index] - mtimes(mtimes(self.skew(self.w_B[uav_index]), self.J_B[uav_index]), self.w_B[uav_index]))

            self.states[uav_index] = vertcat(self.r_I[uav_index], self.v_I[uav_index], self.q[uav_index], self.w_B[uav_index])

            self.dynf[uav_index] = vertcat(dr_I, dv_I, dq, dw)
            self.ctrl[uav_index] = self.T_B[uav_index]

            #set control and state boundary
            if np.asarray(states_bd).size == self.states[uav_index].numel():
                self.states_bd[uav_index] = states_bd
            else:
                self.states_bd[uav_index] = self.states[uav_index].numel() * [-1e10, 1e10]



    def initCost(self, w_F_r=None, w_F_v=None, w_r_formation=None,
                 w_v_formation=None,
                 uav_dist=None, w_uav_collision=None, w_obst=None, w_altitude=50, wthrust=0.1):

        # set the state
        states = self.states
        r_I = [None] * self.uav_num
        v_I = [None] * self.uav_num
        q = [None] * self.uav_num
        w = [None] * self.uav_num

        for i in range(self.uav_num):
            r_I[i] = states[i][0:3]
            v_I[i] = states[i][3:6]
            q[i] = states[i][6:10]
            w[i] = states[i][10:13]

        #set the paramenter
        self.w_F_r = [None] * self.uav_num
        self.w_F_v = [None] * self.uav_num
        self.w_r_formation = [None] * self.uav_num
        self.w_v_formation = [None] * self.uav_num
        self.w_uav_collision = [None] * self.uav_num
        self.cost_auxvar = [None] * self.uav_num
        self.cost_r_formation = [None] * self.uav_num
        self.cost_v_formation = [None] * self.uav_num
        self.uav_dist = [None] * self.uav_num
        self.w_obst = [None] * self.uav_num
        self.cost_altitude = [None] * self.uav_num
        self.cost_r_I = [None] * self.uav_num
        self.cost_v_I = [None] * self.uav_num
        self.cost_collision = [None] * self.uav_num
        self.cost_obst = [None] * self.uav_num
        self.cost_thrust = [None] * self.uav_num
        self.w_altitude = [None] * self.uav_num
        self.path_cost = [None] * self.uav_num
        self.final_cost = [None] * self.uav_num

        for uav_index in range(self.uav_num):
            parameter = []
            if w_F_r is None:
                self.w_F_r[uav_index] = SX.sym('w_F_r' + str(uav_index))
                parameter += [self.w_F_r[uav_index]]
            else:
                self.w_F_r[uav_index] = w_F_r # terminal position cost

            if w_F_v is None:
                self.w_F_v[uav_index] = SX.sym('w_F_v' + str(uav_index))
                parameter += [self.w_F_v[uav_index]]
            else:
                self.w_F_v[uav_index] = w_F_v

            if w_r_formation is None:
                self.w_r_formation[uav_index] = SX.sym('w_r_formation' + str(uav_index))
                parameter += [self.w_r_formation[uav_index]]
            else:
                self.w_r_formation[uav_index] = w_r_formation

            if w_v_formation is None:
                self.w_v_formation[uav_index] = SX.sym('w_v_formation' + str(uav_index))
                parameter += [self.w_v_formation[uav_index]]
            else:
                self.w_v_formation[uav_index] = w_v_formation

            if uav_dist is None:
                self.uav_dist[uav_index] = SX.sym('uav_dist' + str(uav_index))
                parameter += [self.uav_dist[uav_index]]
            else:
                self.uav_dist[uav_index] = uav_dist

            if w_uav_collision is None:
                self.w_uav_collision[uav_index] = SX.sym('w_uav_collision' + str(uav_index))
                parameter += [self.w_uav_collision[uav_index]]
            else:
                self.w_uav_collision[uav_index] = w_uav_collision

            if w_obst is None:
                self.w_obst[uav_index] = SX.sym('w_obst' + str(uav_index))
                parameter += [self.w_obst[uav_index]]
            else:
                self.w_obst[uav_index] = w_obst

            self.cost_auxvar[uav_index] = vcat(parameter)

            # neighbors

            # formation distance and speed costs
            neighbors = [*range(self.uav_num)]
            neighbors.remove(uav_index)
            self.cost_r_formation[uav_index] = 0
            self.cost_v_formation[uav_index] = 0

            # if uav_index == 0:
            #     self.cost_r_formation[uav_index] = norm_2(self.r_I[uav_index] - self.r_I[3] - vcat([2, 2, 0]))
            #     self.cost_v_formation[uav_index] = dot(self.v_I[uav_index] - self.v_I[3], self.v_I[uav_index] - self.v_I[3])
            # if uav_index == self.uav_num-1:
            #     self.cost_r_formation[uav_index] = norm_2(self.r_I[uav_index] - self.r_I[1] - vcat([-2, -2, 0]))
            #     self.cost_v_formation[uav_index] = dot(self.v_I[uav_index] - self.v_I[1], self.v_I[uav_index] - self.v_I[1])

            goal_height = 5
            self.cost_altitude[uav_index] = dot(r_I[uav_index][2] - goal_height, r_I[uav_index][2] - goal_height)

            goal_r_I = np.array([[1, 7, 5], [0.6, 7, 5], [0.3, 7, 5], [1, 7.5, 5]])
            self.cost_r_I[uav_index] = norm_2(r_I[uav_index] - goal_r_I[uav_index] - vcat([self.uav_dist[uav_index], 0, 0]))

            # goal velocity cost
            goal_v_I = np.array([0, 2, 0])
            self.cost_v_I[uav_index] = norm_2(v_I[uav_index] - goal_v_I)

            # collision cost
            self.cost_collision[uav_index] = 0
            for i in neighbors:
                self.cost_collision[uav_index] = self.cost_collision[uav_index] + 1 / fmin(fmax(norm_2(r_I[uav_index] - r_I[i]), 1e-3),
                                                                 0.6) - 1 / 0.6

            self.cost_obst[uav_index] = 5 * fmax(self.wall[0] - r_I[uav_index][0], 0) + 5 * fmax(r_I[uav_index][0] - self.wall[1], 0) + \
                             1 / (fmin(fmax(norm_2(r_I[uav_index][0:2] - self.obstacles[0]), 1e-3), 0.6)) - 1 / 0.6 + \
                             1 / (fmin(fmax(norm_2(r_I[uav_index][0:2] - self.obstacles[1]), 1e-3), 0.6)) - 1 / 0.6

            # the thrust cost
            self.cost_thrust[uav_index] = dot(self.T_B[uav_index], self.T_B[uav_index])

            self.w_altitude[uav_index] = w_altitude

            self.path_cost[uav_index] = self.w_r_formation[uav_index] * self.cost_r_formation[uav_index] + \
                            self.w_v_formation[uav_index] * self.cost_v_formation[uav_index] + \
                            self.w_uav_collision[uav_index] * self.cost_collision[uav_index] + \
                            self.w_altitude[uav_index] * self.cost_altitude[uav_index] + \
                            10 * self.w_obst[uav_index] * self.cost_obst[uav_index] + \
                            wthrust * self.cost_thrust[uav_index]
            # self.w_v_formation * self.cost_v_formation + \
            # self.w_F_r * self.cost_r_I + \
            # self.w_F_v * self.cost_v_I + \

            self.final_cost[uav_index] = self.w_F_r[uav_index] * self.cost_r_I[uav_index] + \
                            self.w_F_v[uav_index] * self.cost_v_I[uav_index] + \
                            100 * self.w_r_formation[uav_index] * self.cost_r_formation[uav_index] + \
                            100 * self.w_v_formation[uav_index] * self.cost_v_formation[uav_index] + \
                            self.w_altitude[uav_index] * self.cost_altitude[uav_index] + \
                            self.w_uav_collision[uav_index] * self.cost_collision[uav_index]
            print(self.final_cost)


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

    def play_multi_animation(self, wing_len, state_traj, state_traj_ref=None, dt=None, save_option=0,
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
        position = [None] * self.uav_num
        position_ref = [None] * self.uav_num
        for i in range(self.uav_num):
            position[i] = self.get_UAV_position(wing_len, state_traj[i])
            if state_traj_ref is None:
                position_ref[i] = self.get_UAV_position(0, numpy.zeros_like(position[i]))
            else:
                position_ref[i] = self.get_UAV_position(wing_len, state_traj_ref[i])

        # animation
        line_traj = [None] * self.uav_num
        c_x = [None] * self.uav_num
        c_y = [None] * self.uav_num
        c_z = [None] * self.uav_num
        r1_x = [None] * self.uav_num
        r1_y = [None] * self.uav_num
        r1_z = [None] * self.uav_num
        r2_x = [None] * self.uav_num
        r2_y = [None] * self.uav_num
        r2_z = [None] * self.uav_num
        r3_x = [None] * self.uav_num
        r3_y = [None] * self.uav_num
        r3_z = [None] * self.uav_num
        r4_x = [None] * self.uav_num
        r4_y = [None] * self.uav_num
        r4_z = [None] * self.uav_num
        line_arm1 = [None] * self.uav_num
        line_arm2 = [None] * self.uav_num
        line_arm3 = [None] * self.uav_num
        line_arm4 = [None] * self.uav_num
        line_traj_ref = [None] * self.uav_num
        for i in range(self.uav_num):
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
            for i in range(self.uav_num):
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
    def __init__(self, uav_num, obstacles, wall, project_name='my UAV swarm 2D'):
        self.project_name = 'my uav swarm 2D'
        self.uav_num = uav_num
        self.obstacles = obstacles
        self.wall = wall

    def initDyn(self, Jxy=None, mass=None, l=0.1, states_bd=[None, None],
                ctrl_bd=[None, None], dt=1e-3):

        self.Jxy = [None] * self.uav_num
        self.mass = [None] * self.uav_num
        self.l = [None] * self.uav_num
        self.dyn_auxvar = [None] * self.uav_num
        self.J_B = [None] * self.uav_num
        self.m = [None] * self.uav_num
        self.thrust_B = [None] * self.uav_num
        self.M_B = [None] * self.uav_num
        self.T_B = [None] * self.uav_num
        self.dynf = [None] * self.uav_num
        self.ctrl = [None] * self.uav_num
        self.states = [None] * self.uav_num
        self.states_bd = [None] * self.uav_num
        self.ctrl_bd = [None] * self.uav_num

        self.r_I = [None] * self.uav_num
        self.v_I = [None] * self.uav_num
        self.w_B = [None] * self.uav_num
        self.theta = [None] * self.uav_num

        # define the state of the uavs
        # Model reference : http://underactuated.mit.edu/acrobot.html
        for uav_index in range(self.uav_num):
            rx, ry = SX.sym('rx' + str(uav_index)), SX.sym('ry' + str(uav_index))
            self.r_I[uav_index] = vertcat(rx, ry)
            vx, vy = SX.sym('vx' + str(uav_index)), SX.sym('vy' + str(uav_index))
            self.v_I[uav_index] = vertcat(vx, vy)

            self.theta[uav_index] = SX.sym('theta' + str(uav_index))
            self.w_B[uav_index] = SX.sym('w' + str(uav_index))
            # define the Planar quadrotor input
            f1, f2 = SX.sym('f1' + str(uav_index)), SX.sym('f2' + str(uav_index))
            self.T_B[uav_index] = vertcat(f1, f2)

            # global parameter
            g = 10

            # parameters settings
            parameter = []
            if Jxy is None:
                self.Jxy[uav_index] = SX.sym('Jxy' + str(uav_index))
                parameter += [self.Jxy[uav_index]]
            else:
                self.Jxy[uav_index] = Jxy

            if mass is None:
                self.mass[uav_index] = SX.sym('mass' + str(uav_index))
                parameter += [self.mass[uav_index]]
            else:
                self.mass[uav_index] = mass

            if l is None:
                self.l[uav_index] = SX.sym('l' + str(uav_index))
                parameter += [self.l[uav_index]]
            else:
                self.l[uav_index] = l

            self.dyn_auxvar[uav_index] = vcat(parameter)

            # Angular moment of inertia
            # self.J_B = diag(vertcat(self.Jx, self.Jy))
            # Gravity
            self.g_I = vertcat(0, -g)
            # Mass of rocket, assume is little changed during the landing process
            self.m[uav_index] = self.mass[uav_index]

            dr_I = self.v_I[uav_index]
            dv_I = 1 / self.m[uav_index] * vertcat(-sin(self.theta[uav_index]) * (self.T_B[uav_index][0] + self.T_B[uav_index][1]),
                                        cos(self.theta[uav_index]) * (self.T_B[uav_index][0] + self.T_B[uav_index][1])) + self.g_I
            dtheta = self.w_B[uav_index]
            dw_B = 1 / self.Jxy[uav_index] * self.l[uav_index] * (self.T_B[uav_index][0] - self.T_B[uav_index][1])

            self.states[uav_index] = vertcat(self.r_I[uav_index], self.v_I[uav_index], self.theta[uav_index], self.w_B[uav_index])

            if np.asarray(states_bd).size == self.states[uav_index].numel():
                self.states_bd[uav_index] = states_bd
            else:
                self.states_bd[uav_index] = self.states[uav_index].numel() * [-1e10, 1e10]

            self.ctrl[uav_index] = self.T_B[uav_index]

            self.dynf[uav_index] = vertcat(dr_I, dv_I, dtheta, dw_B)
            self.dt = dt

    def initCost(self, adj_matrix=None,w_F_r=None, w_F_v=None, w_r_formation=None, w_v_formation=None,
                 uav_dist=None, w_uav_collision=None, w_obst=None, wthrust=0.1, w_waypoint=None, goal_r_I=None):

        # set the state
        states = self.states
        r_I = [None] * self.uav_num
        v_I = [None] * self.uav_num
        q = [None] * self.uav_num
        w = [None] * self.uav_num

        for i in range(self.uav_num):
            r_I[i] = states[i][0:2]
            v_I[i] = states[i][2:4]
            q[i] = states[i][4]
            w[i] = states[i][5]

        # set the paramenter
        self.waypoint = [None] * self.uav_num
        self.w_F_r = [None] * self.uav_num
        self.w_F_v = [None] * self.uav_num
        self.w_r_formation = [None] * self.uav_num
        self.w_v_formation = [None] * self.uav_num
        self.w_uav_collision = [None] * self.uav_num
        self.w_obst = [None] * self.uav_num
        self.w_waypoint = [None] * self.uav_num

        self.cost_auxvar = [None] * self.uav_num
        self.cost_r_formation = [None] * self.uav_num
        self.cost_v_formation = [None] * self.uav_num
        self.uav_dist = [None] * self.uav_num
        self.cost_waypoint = [None] * self.uav_num

        self.goal_r_I = [None] * self.uav_num
        self.cost_r_I = [None] * self.uav_num
        self.cost_v_I = [None] * self.uav_num
        self.cost_collision = [None] * self.uav_num
        self.cost_obst = [None] * self.uav_num
        self.cost_thrust = [None] * self.uav_num

        self.path_cost = [None] * self.uav_num
        self.final_cost = [None] * self.uav_num

        for uav_index in range(self.uav_num):
            parameter = []
            if w_F_r is None:
                self.w_F_r[uav_index] = SX.sym('w_F_r' + str(uav_index))
                parameter += [self.w_F_r[uav_index]]
            else:
                self.w_F_r[uav_index] = w_F_r

            if w_F_v is None:
                self.w_F_v[uav_index] = SX.sym('w_F_v' + str(uav_index))
                parameter += [self.w_F_v[uav_index]]
            else:
                self.w_F_v[uav_index] = w_F_v

            if w_r_formation is None:
                self.w_r_formation[uav_index] = SX.sym('w_r_formation' + str(uav_index))
                parameter += [self.w_r_formation[uav_index]]
            else:
                self.w_r_formation[uav_index] = w_r_formation

            if w_v_formation is None:
                self.w_v_formation[uav_index] = SX.sym('w_v_formation' + str(uav_index))
                parameter += [self.w_v_formation[uav_index]]
            else:
                self.w_v_formation[uav_index] = w_v_formation

            if uav_dist is None:
                self.uav_dist[uav_index] = SX.sym('uav_dist' + str(uav_index))
                parameter += [self.uav_dist[uav_index]]
            else:
                self.uav_dist[uav_index] = uav_dist

            if w_uav_collision is None:
                self.w_uav_collision[uav_index] = SX.sym('w_uav_collision' + str(uav_index))
                parameter += [self.w_uav_collision[uav_index]]
            else:
                self.w_uav_collision[uav_index] = w_uav_collision

            if w_obst is None:
                self.w_obst[uav_index] = SX.sym('w_obst' + str(uav_index))
                parameter += [self.w_obst[uav_index]]
            else:
                self.w_obst[uav_index] = w_obst

            if w_waypoint is None:
                self.w_waypoint[uav_index] = SX.sym('w_waypoint' + str(uav_index))
                parameter += [self.w_waypoint[uav_index]]
            else:
                self.w_waypoint[uav_index] = w_waypoint

            self.cost_auxvar[uav_index] = vcat(parameter)
            waypoint_x, waypoint_y = SX.sym('waypoint_x' + str(uav_index)), SX.sym('waypoint_y' + str(uav_index))
            self.waypoint[uav_index] = vertcat(waypoint_x, waypoint_y)
            # neighbors

            # formation distance and speed costs
            if adj_matrix[uav_index]:
                neighbors = list(np.where(np.array(adj_matrix[uav_index]) == 1)[0])
            else:
                neighbors = [*range(num_uav)]
            neighbors.remove(uav_index)

            self.cost_r_formation[uav_index] = 0
            self.cost_v_formation[uav_index] = 0
            self.cost_v_formation[uav_index] += dot(self.v_I[1] - 2, self.v_I[1] - 2)

            for j in neighbors:
                self.cost_r_formation[uav_index] += dot(self.r_I[uav_index] - r_I[j] - vcat([(uav_index - j) * self.uav_dist[uav_index], 0]),
                                             self.r_I[uav_index] - r_I[j] - vcat([(uav_index - j) * self.uav_dist[uav_index], 0]))
                self.cost_v_formation[uav_index] += dot(self.v_I[uav_index] - v_I[j], self.v_I[uav_index] - v_I[j])

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
            self.goal_r_I = goal_r_I
            self.cost_r_I[uav_index] = dot(self.r_I[uav_index] - self.goal_r_I[uav_index] - vcat([uav_index * self.uav_dist[uav_index], 0]),
                                self.r_I[uav_index] - self.goal_r_I[uav_index] - vcat([uav_index * self.uav_dist[uav_index], 0]))

            # else:
            #    self.cost_r_I =0

            #    self.cost_r_I = 0
            # goal velocity cost
            goal_v_I = np.array([0, 2])
            self.cost_v_I[uav_index] = dot(self.v_I[uav_index] - goal_v_I, self.v_I[uav_index] - goal_v_I)

            # waypoint cost
            self.cost_waypoint[uav_index] = dot(self.r_I[uav_index]-self.waypoint[uav_index], self.r_I[uav_index]-self.waypoint[uav_index])

            # collision cost
            self.cost_collision[uav_index] = 0
            for i in neighbors:
                # self.cost_collision = self.cost_collision + 1 / fmin(fmax(norm_2(self.r_I - uavswarm[i].r_I), 1e-3),
                #                                                     0.6) - 1 / 0.6
                self.cost_collision[uav_index] = self.cost_collision[uav_index] + 1 / dot(self.r_I[uav_index] - r_I[i], self.r_I[uav_index] - r_I[i])

            self.cost_obst[uav_index] = 5 * fmax(self.wall[0] - self.r_I[uav_index][0], 0) + 5 * fmax(self.r_I[uav_index][0] - self.wall[1], 0) + \
                            1 / (fmin(fmax(norm_2(self.r_I[uav_index][0:2] - self.obstacles[0]), 1e-6), 0.6)) - 1 / 0.6 + \
                            1 / (fmin(fmax(norm_2(self.r_I[uav_index][0:2] - self.obstacles[1]), 1e-6), 0.6)) - 1 / 0.6

            # the thrust cost
            self.cost_thrust[uav_index] = dot(self.T_B[uav_index], self.T_B[uav_index])

            self.path_cost[uav_index] = self.w_r_formation[uav_index] * self.cost_r_formation[uav_index] + \
                             self.w_v_formation[uav_index] * self.cost_v_formation[uav_index] + \
                             self.w_uav_collision[uav_index] * self.cost_collision[uav_index] + \
                             self.w_obst[uav_index] * self.cost_obst[uav_index] + \
                             self.w_waypoint[uav_index] * self.cost_waypoint[uav_index] + \
                             wthrust[uav_index] * self.cost_thrust[uav_index]
            # self.w_v_formation * self.cost_v_formation + \
            # self.w_F_r * self.cost_r_I + \
            # self.w_F_v * self.cost_v_I + \

            self.final_cost[uav_index] = self.w_F_r[uav_index] * self.cost_r_I[uav_index] + \
                              self.w_F_v[uav_index] * self.cost_v_I[uav_index] + \
                              10 * self.w_r_formation[uav_index] * self.cost_r_formation[uav_index] + \
                              10 * self.w_v_formation[uav_index] * self.cost_v_formation[uav_index] + \
                              self.w_uav_collision[uav_index] * self.cost_collision[uav_index]

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

    def play_multi_animation(self, wing_len, state_traj, state_traj_ref=None, dt=None, save_option=0,
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
        position = [None] * self.uav_num
        position_ref = [None] * self.uav_num
        for i in range(self.uav_num):
            position[i] = self.get_UAV_position(wing_len, state_traj[i])
            if state_traj_ref is None:
                position_ref[i] = self.get_UAV_position(0, numpy.zeros_like(position[i]))
            else:
                position_ref[i] = self.get_UAV_position(wing_len, state_traj_ref[i])

        # animation
        line_traj = [None] * self.uav_num
        c_x = [None] * self.uav_num
        c_y = [None] * self.uav_num

        r1_x = [None] * self.uav_num
        r1_y = [None] * self.uav_num

        r2_x = [None] * self.uav_num
        r2_y = [None] * self.uav_num

        line_arm1 = [None] * self.uav_num
        line_arm2 = [None] * self.uav_num

        line_traj_ref = [None] * self.uav_num
        for i in range(self.uav_num):
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

    def play_multi_animation_3(self, wing_len, state_traj, state_traj_ref=None, dt=None, save_option=1,
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
        risky_width = self.obstacles[1][0] - self.obstacles[0][0] + 0.2
        risky_height = self.obstacles[1][1] - self.obstacles[0][1] + 0.2
        risky = plt.Rectangle((self.obstacles[0][0], self.obstacles[0][1]), risky_width, risky_height, color='r', alpha=0.5)

        ax.add_patch(wall1)
        ax.add_patch(wall2)
        ax.add_patch(risky)

        # data
        position = [None] * self.uav_num
        position_ref = [None] * self.uav_num
        for i in range(self.uav_num):
            position[i] = self.get_UAV_position(wing_len, state_traj[i])
            if state_traj_ref is None:
                position_ref[i] = self.get_UAV_position(0, numpy.zeros_like(position[i]))
            else:
                position_ref[i] = self.get_UAV_position(wing_len, state_traj_ref[i])

        # animation
        line_traj = [None] * self.uav_num
        c_x = [None] * self.uav_num
        c_y = [None] * self.uav_num

        r1_x = [None] * self.uav_num
        r1_y = [None] * self.uav_num

        r2_x = [None] * self.uav_num
        r2_y = [None] * self.uav_num

        line_arm1 = [None] * self.uav_num
        line_arm2 = [None] * self.uav_num

        line_traj_ref = [None] * self.uav_num
        for i in range(self.uav_num):
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
            for i in range(self.uav_num):
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

            return line_traj[0], line_arm1[0], line_arm2[0], line_traj[1], line_arm1[1], line_arm2[1], time_text  # , \

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