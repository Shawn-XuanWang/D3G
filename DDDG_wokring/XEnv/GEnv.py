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
        self.UAV_labels = None
        self.r_I = [None] * self.uav_num
        self.v_I = [None] * self.uav_num
        self.w_B = [None] * self.uav_num
        self.q_I = [None] * self.uav_num
        self.R = [None] * self.uav_num
        self.states = [None] * uav_num


        #obstacles
        self.obstacles = [[4.3, 3], [4.8, 3]]
        self.wall = [0, 5]

    def setRandomObstacles(self, n_obstacle=3, ball_r=1.0):

        self.obstacle_plot_info = []
        self.obstacle_info = []
        for i in range(n_obstacle):
            # ball position
            ball_xy = np.random.uniform(low=np.array([-5.0, -5.0]) + ball_r,
                                         high=np.array([5.0, 5.0]) - ball_r,
                                         size=(3,))
            self.obstacle_info.append([ball_xy, ball_r])

            # ball surface
            self.obstacle_plot_info.append(self._3DBall_surface(ball_xy, ball_r))



    # allow you set the random obstacles
    def initDyn(self, c=None, states_bd=[None, None],
                ctrl_bd=[None, None], dt=1e-3):
        # define the parameter
        self.c = [None] * self.uav_num
        self.dynf = [None] * self.uav_num
        self.ctrl = [None] * self.uav_num
        self.states_bd = [None] * self.uav_num
        self.ctrl_bd = [None] * self.uav_num

        # global parameter
        self.dt = dt

        # set parameter for each uav
        for uav_index in range(self.uav_num):
            parameter = []

            # define the position
            rx, ry = SX.sym('rx' + str(uav_index)), SX.sym('ry' + str(uav_index))
            self.r_I[uav_index] = vertcat(rx, ry)

            vx, vy = SX.sym('vx' + str(uav_index)), SX.sym('vy' + str(uav_index))
            # self.v_I[uav_index] = vertcat(vx, vy)

            # define the orientation
            self.q_I[uav_index] = SX.sym('theta' + str(uav_index))
            self.w_B[uav_index] = SX.sym('w' + str(uav_index))

            # define the input
            u = SX.sym('u' + str(uav_index))
            vx, vy = u * math.cos(self.q_I[uav_index]), u * math.sin(self.q_I[uav_index])
            self.v_I[uav_index] = vertcat(vx, vy)

            # 2D rotation
            self.R[uav_index] = self.angle2rot(self.q_I[uav_index])
            C_I_B = self.R[uav_index]  # inertial to body


            # define the state
            self.states[uav_index] = vertcat(self.r_I[uav_index], self.q_I[uav_index])

            # Newton's law
            dr_I = self.v_I[uav_index]
            dq_I = self.w_B[uav_index]

            self.ctrl[uav_index] = vertcat(u, self.w_B[uav_index])

            # set control and state boundary
            if np.asarray(states_bd).size == self.states[uav_index].numel():
                self.states_bd[uav_index] = states_bd
            else:
                self.states_bd[uav_index] = self.states[uav_index].numel() * [-1e10, 1e10]

            self.dynf[uav_index] = vertcat(dr_I, dq_I)

    def initCost(self, w_F_r=None, w_r_formation=None,
                 uav_dist=None, w_uav_collision=None, w_obst=None):

        # set the state
        states = self.states
        r_I = [None] * self.uav_num
        v_I = [None] * self.uav_num
        q = [None] * self.uav_num

        for i in range(self.uav_num):
            r_I[i] = states[i][0:2]
            q[i] = states[i][2]

        #set the paramenter
        self.w_F_r = [None] * self.uav_num
        self.w_r_formation = [None] * self.uav_num
        self.w_uav_collision = [None] * self.uav_num
        self.cost_auxvar = [None] * self.uav_num
        self.cost_r_formation = [None] * self.uav_num
        self.uav_dist = [None] * self.uav_num
        self.w_obst = [None] * self.uav_num
        self.cost_altitude = [None] * self.uav_num
        self.cost_r_I = [None] * self.uav_num
        self.cost_collision = [None] * self.uav_num
        self.cost_obst = [None] * self.uav_num
        self.path_cost = [None] * self.uav_num
        self.final_cost = [None] * self.uav_num

        for uav_index in range(self.uav_num):
            parameter = []
            if w_F_r is None:
                self.w_F_r[uav_index] = SX.sym('w_F_r' + str(uav_index))
                parameter += [self.w_F_r[uav_index]]
            else:
                self.w_F_r[uav_index] = w_F_r # terminal position cost

            if w_r_formation is None:
                self.w_r_formation[uav_index] = SX.sym('w_r_formation' + str(uav_index))
                parameter += [self.w_r_formation[uav_index]]
            else:
                self.w_r_formation[uav_index] = w_r_formation


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

            # if uav_index == 0:
            #     self.cost_r_formation[uav_index] = norm_2(self.r_I[uav_index] - self.r_I[3] - vcat([2, 2, 0]))
            #     self.cost_v_formation[uav_index] = dot(self.v_I[uav_index] - self.v_I[3], self.v_I[uav_index] - self.v_I[3])
            # if uav_index == self.uav_num-1:
            #     self.cost_r_formation[uav_index] = norm_2(self.r_I[uav_index] - self.r_I[1] - vcat([-2, -2, 0]))
            #     self.cost_v_formation[uav_index] = dot(self.v_I[uav_index] - self.v_I[1], self.v_I[uav_index] - self.v_I[1])

            goal_r_I = np.array([1, 7])
            self.cost_r_I[uav_index] = norm_2(r_I[uav_index] - goal_r_I - vcat([self.uav_dist[uav_index], 0]))

            # collision cost
            self.cost_collision[uav_index] = 0
            for i in neighbors:
                self.cost_collision[uav_index] = self.cost_collision[uav_index] + 1 / fmin(fmax(norm_2(r_I[uav_index] - r_I[i]), 1e-3),
                                                                 0.6) - 1 / 0.6

            self.cost_obst[uav_index] = 5 * fmax(self.wall[0] - r_I[uav_index][0], 0) + 5 * fmax(r_I[uav_index][0] - self.wall[1], 0) + \
                             1 / (fmin(fmax(norm_2(r_I[uav_index][0:2] - self.obstacles[0]), 1e-3), 0.6)) - 1 / 0.6 + \
                             1 / (fmin(fmax(norm_2(r_I[uav_index][0:2] - self.obstacles[1]), 1e-3), 0.6)) - 1 / 0.6

            # the thrust cost
            self.path_cost[uav_index] = self.w_r_formation[uav_index] * self.cost_r_formation[uav_index] + \
                            self.w_uav_collision[uav_index] * self.cost_collision[uav_index] + \
                            10 * self.w_obst[uav_index] * self.cost_obst[uav_index]
            # self.w_v_formation * self.cost_v_formation + \
            # self.w_F_r * self.cost_r_I + \
            # self.w_F_v * self.cost_v_I + \

            self.final_cost[uav_index] = self.w_F_r[uav_index] * self.cost_r_I[uav_index] + \
                            100 * self.w_r_formation[uav_index] * self.cost_r_formation[uav_index] + \
                            self.w_uav_collision[uav_index] * self.cost_collision[uav_index]
            print(self.final_cost)

    def play_multi_animation(self, state_traj, state_traj_ref=None, dt=None, save_option=0,
                             title='UAV Maneuvering'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_zlim(0, 10)
        ax.set_ylim(-1, 10)
        ax.set_box_aspect(aspect=(7, 11))
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


    def angle2rot(self,angle):
        R = vertcat(
            horzcat(math.cos(angle),math.sin(angle)),
            horzcat(-math.sin(angle), math.cos(angle))
        )
        return R
