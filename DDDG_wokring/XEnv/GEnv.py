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
        self.q_I = [None] * self.uav_num
        self.states = [None] * self.uav_num


        #obstacles
        self.obstacles = [[2.5, 0], [2.5, 5]]
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
    def initDyn(self, states_bd=[None, None],
                ctrl_bd=[None, None], dt=1e-3):
        # define the parameter
        self.dynf = [None] * self.uav_num
        self.ctrl = [None] * self.uav_num
        self.states_bd = [None] * self.uav_num
        self.ctrl_bd = [None] * self.uav_num

        # global parameter
        self.dt = dt

        # set parameter for each uav
        for uav_index in range(self.uav_num):
            # define the position
            rx, ry = SX.sym('rx' + str(uav_index)), SX.sym('ry' + str(uav_index))
            self.r_I[uav_index] = vertcat(rx, ry)

            # define the orientation
            q = SX.sym('theta' + str(uav_index))
            self.q_I[uav_index] = q

            # define the input
            u_v = SX.sym('v' + str(uav_index))
            u_w = SX.sym('w' + str(uav_index))

            # define the state
            self.states[uav_index] = vertcat(self.r_I[uav_index], self.q_I[uav_index])

            # Newton's law
            # dr_I = vertcat(u_v * (1-0.5*self.q_I[uav_index]*self.q_I[uav_index]), u_v * self.q_I[uav_index])
            dr_I = vertcat(u_v * cos(self.q_I[uav_index]), u_v * sin(self.q_I[uav_index]))
            dq_I = u_w

            self.ctrl[uav_index] = vertcat(u_v, u_w)

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

            goal_r_I = np.array([4, 0.5])
            self.cost_r_I[uav_index] = norm_2(r_I[uav_index] - goal_r_I - vcat([self.uav_dist[uav_index], 0]))

            # collision cost
            self.cost_collision[uav_index] = 0
            for i in neighbors:
                self.cost_collision[uav_index] = self.cost_collision[uav_index] + 1 / fmin(fmax(norm_2(r_I[uav_index] - r_I[i]), 1e-3),
                                                                 0.6) - 1 / 0.6

            # self.cost_obst[uav_index] = 5 * fmax(self.wall[0] - r_I[uav_index][0], 0) + 5 * fmax(r_I[uav_index][0] - self.wall[1], 0) + \
            #                  1 / (fmin(fmax(norm_2(r_I[uav_index][0:2] - self.obstacles[0]), 1e-3), 0.6)) - 1 / 0.6 + \
            #                  1 / (fmin(fmax(norm_2(r_I[uav_index][0:2] - self.obstacles[1]), 1e-3), 0.6)) - 1 / 0.6

            # distance between robot and obstacle
            vec_obst = [self.obstacles[1][0] - self.obstacles[0][0],self.obstacles[1][1] - self.obstacles[0][1]]
            vec_r_I = [self.obstacles[1][0] - r_I[uav_index][0],self.obstacles[1][1] - r_I[uav_index][1]]
            r_obst = (vec_obst[0]*vec_r_I[0]+vec_obst[1]*vec_r_I[1])/norm_2(vec_obst) * vec_obst/norm_2(vec_obst)
            r_obst = [vec_r_I[0] - r_obst[0], vec_r_I[1] - r_obst[1]]
            r_obst = np.dot(r_obst,r_obst)

            # risk avoidance cost
            self.cost_obst[uav_index] = 5 * fmax(self.wall[0] - r_I[uav_index][0], 0) + 5 * fmax(r_I[uav_index][0] - self.wall[1], 0) + \
                            1 / fmax(r_obst,0.1)

            self.path_cost[uav_index] = 100*self.w_r_formation[uav_index] * self.cost_r_formation[uav_index] + \
                            self.w_uav_collision[uav_index] * self.cost_collision[uav_index] + \
                            10 * self.w_obst[uav_index] * self.cost_obst[uav_index]
            # self.w_v_formation * self.cost_v_formation + \
            # self.w_F_r * self.cost_r_I + \
            # self.w_F_v * self.cost_v_I + \

            self.final_cost[uav_index] = self.w_F_r[uav_index] * self.cost_r_I[uav_index] + \
                            1 * self.w_r_formation[uav_index] * self.cost_r_formation[uav_index] + \
                            self.w_uav_collision[uav_index] * self.cost_collision[uav_index]
            print(self.final_cost)

    def get_UAV_position(self, state_traj):
        # horizon
        horizon = np.size(state_traj, 0)
        position = np.zeros((horizon, 3))
        for t in range(horizon):
            # position of COM
            rc = state_traj[t, 0:2]
            # altitude of quaternion
            q = state_traj[t, 2]

            # store
            position[t, 0:2] = rc
            position[t, 2] = q

        return position

    def play_multi_animation(self, state_traj, state_traj_ref=None, dt=None, save_option=0,
                             title='UAV Maneuvering'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
        ax.set_ylim(-1, 10)
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
            position[i] = self.get_UAV_position(state_traj[i])
            if state_traj_ref is None:
                position_ref[i] = self.get_UAV_position(numpy.zeros_like(position[i]))
            else:
                position_ref[i] = self.get_UAV_position(state_traj_ref[i])

        # animation
        line_traj = [None] * self.uav_num
        c_x = [None] * self.uav_num
        c_y = [None] * self.uav_num
        line_traj_ref = [None] * self.uav_num
        for i in range(self.uav_num):
            line_traj[i], = ax.plot(position[i][:1, 0], position[i][:1, 1])
            c_x[i], c_y[i] = position[i][0, 0:2]

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
            for i in range(self.uav_num):
                line_traj[i].set_data(position[i][:num, 0], position[i][:num, 1])

                # uav
                c_x[i], c_y[i] = position[i][num, 0:2]


            return line_traj[0], line_traj[1], line_traj[2], line_traj[3], time_text,


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
