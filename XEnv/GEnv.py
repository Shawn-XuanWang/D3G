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
    def __init__(self, uav_num, obstacles, wall_vet, wall_hor, goal_r_I,project_name='my UAV swarm'):
        self.project_name = 'my uav swarm'
        self.uav_num = uav_num
        self.UAV_labels = None
        self.r_I = [None] * self.uav_num
        self.v_I = [None] * self.uav_num
        self.q_I = [None] * self.uav_num
        self.states = [None] * self.uav_num
        self.goal_r_I = goal_r_I


        #obstacles
        self.obstacles = obstacles
        self.wall_vet = wall_vet
        self.wall_hor = wall_hor

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

            # define the speed
            vx, vy = SX.sym('vx' + str(uav_index)), SX.sym('vy' + str(uav_index))
            self.v_I[uav_index] = vertcat(vx, vy)

            # define the input
            u_a = SX.sym('a' + str(uav_index))
            u_w = SX.sym('w' + str(uav_index))

            # define the state
            self.states[uav_index] = vertcat(self.r_I[uav_index], self.q_I[uav_index], self.v_I[uav_index])

            # Newton's law
            # dr_I = vertcat(u_v * (1-0.5*self.q_I[uav_index]*self.q_I[uav_index]), u_v * self.q_I[uav_index])
            dr_I = self.v_I[uav_index]
            dq_I = u_w
            dv_I = vertcat(u_a * cos(self.q_I[uav_index]), u_a * sin(self.q_I[uav_index]))

            self.ctrl[uav_index] = vertcat(u_a, u_w)

            # set control and state boundary
            if np.asarray(states_bd).size == self.states[uav_index].numel():
                self.states_bd[uav_index] = states_bd
            else:
                self.states_bd[uav_index] = self.states[uav_index].numel() * [-1e10, 1e10]

            self.dynf[uav_index] = vertcat(dr_I, dq_I, dv_I)

    def initCost(self, adj_matrix=None, w_F_r=None, w_F_v=None, w_r_formation=None, w_v_formation=None,
                 uav_dist=None, w_uav_collision=None, w_obst=None,w_waypoint=None):

        # set the state
        states = self.states
        r_I = [None] * self.uav_num
        v_I = [None] * self.uav_num
        q = [None] * self.uav_num

        for i in range(self.uav_num):
            r_I[i] = states[i][0:2]
            q[i] = states[i][2]
            v_I[i] = states[i][3:5]

        #set the paramenter
        self.waypoint = [None] * self.uav_num
        self.w_F_r = [None] * self.uav_num
        self.w_F_v = [None] * self.uav_num
        self.w_r_formation = [None] * self.uav_num
        self.w_v_formation = [None] * self.uav_num
        self.w_uav_collision = [None] * self.uav_num
        self.w_waypoint = [None] * self.uav_num
        self.cost_auxvar = [None] * self.uav_num
        self.cost_r_formation = [None] * self.uav_num
        self.cost_v_formation = [None] * self.uav_num
        self.cost_waypoint = [None] * self.uav_num
        self.uav_dist = [None] * self.uav_num
        self.w_obst = [None] * self.uav_num
        self.cost_r_I = [None] * self.uav_num
        self.cost_v_I = [None] * self.uav_num
        self.cost_collision = [None] * self.uav_num
        self.cost_obst = [None] * self.uav_num
        self.path_cost = [None] * self.uav_num
        self.final_cost = [None] * self.uav_num

        self.obst_d = [None] * self.uav_num

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
                self.w_F_v[uav_index] = w_F_v # terminal position cost

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

            waypoint_x, waypoint_y= SX.sym('waypoint_x' + str(uav_index)), SX.sym('waypoint_y' + str(uav_index))
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

            for j in neighbors:
                # maintain distance and orientation
                self.cost_r_formation[uav_index] += dot(self.r_I[uav_index] - r_I[j] - vcat([0, (uav_index - j) * self.uav_dist[uav_index]]),
                                             self.r_I[uav_index] - r_I[j] - vcat([0, (uav_index - j) * self.uav_dist[uav_index]]))

                self.cost_v_formation[uav_index] += dot(self.v_I[uav_index] - v_I[j], self.v_I[uav_index] - v_I[j])



            self.cost_r_I[uav_index] = dot(r_I[uav_index] - self.goal_r_I[uav_index], r_I[uav_index] - self.goal_r_I[uav_index])

            goal_v_I = 2.5
            self.cost_v_I[uav_index] = dot(v_I[uav_index] - goal_v_I, v_I[uav_index] - goal_v_I)

            # waypoint cost
            if uav_index == 1:
                self.cost_waypoint[uav_index] = dot(self.r_I[uav_index] - self.waypoint[uav_index][0:2], self.r_I[uav_index] - self.waypoint[uav_index][0:2])
            else:
                self.cost_waypoint[uav_index] = dot(self.r_I[uav_index] - self.waypoint[uav_index][0:2] - vcat([0, (uav_index - 1) * self.uav_dist[uav_index]]),
                                                    self.r_I[uav_index] - self.waypoint[uav_index][0:2] - vcat([0, (uav_index - 1) * self.uav_dist[uav_index]]))

            # collision cost
            self.obst_d[uav_index] = self.obstacle_dis(self.obstacles, self.r_I[uav_index])

            # print(self.obst_d[uav_index])
            self.cost_collision[uav_index] = 0
            for i in neighbors:
                self.cost_collision[uav_index] = self.cost_collision[uav_index] + 1 / fmin(fmax(norm_2(r_I[uav_index] - r_I[i]), 1e-3),
                                                                 0.6) - 1 / 0.6
            sen_range_1 = 0.25
            sen_range_2 = 0.5
            self.cost_obst[uav_index] = (sen_range_1 / fmin(dot(self.r_I[uav_index][1] - self.wall_hor[0], self.r_I[uav_index][1] - self.wall_hor[0]),
                                                sen_range_1) - 1) * dot(self.r_I[uav_index] - self.r_I[1], self.r_I[uav_index] - self.r_I[1]) + \
                                        (sen_range_2 / fmin(dot(self.r_I[uav_index] - self.obstacles[1], self.r_I[uav_index] - self.obstacles[1]),
                                                sen_range_2) - 1) * dot(self.r_I[uav_index] - self.r_I[1], self.r_I[uav_index] - self.r_I[1]) + \
                                        5 * fmax(self.wall_vet[0] - r_I[uav_index][0], 0) + 5 * fmax(r_I[uav_index][0] - self.wall_vet[1], 0)

            self.path_cost[uav_index] = self.w_r_formation[uav_index] * self.cost_r_formation[uav_index] + \
                            self.w_v_formation[uav_index] * self.cost_v_formation[uav_index] + \
                            1 * self.w_uav_collision[uav_index] * self.cost_collision[uav_index] + \
                            self.w_waypoint[uav_index] * self.cost_waypoint[uav_index] + \
                            10 * self.w_obst[uav_index] * self.cost_obst[uav_index]
            # self.w_v_formation * self.cost_v_formation + \
            # self.w_F_r * self.cost_r_I + \
            # self.w_F_v * self.cost_v_I + \

            self.final_cost[uav_index] = self.w_F_r[uav_index] * self.cost_r_I[uav_index] + \
                            1 * self.w_F_v[uav_index] * self.cost_v_I[uav_index] + \
                            1 * self.w_r_formation[uav_index] * self.cost_r_formation[uav_index] + \
                            1 * self.w_v_formation[uav_index] * self.cost_v_formation[uav_index] + \
                            1 * self.w_uav_collision[uav_index] * self.cost_collision[uav_index]
            # print(self.final_cost)

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

    def play_multi_animation(self, state_traj, state_traj_ref=None, dt=None, save_option=0, waypoints = None,
                             title='UAV Maneuvering'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('X (m)', fontsize=18, labelpad=5)
        ax.set_ylabel('Y (m)', fontsize=18, labelpad=5)
        ax.set_ylim(-1, 6)
        ax.set_title(title, pad=20, fontsize=15)

        wall1 = plt.Rectangle((0, -1), 0.1, 11, color='limegreen', alpha=0.2)
        wall2 = plt.Rectangle((5, -1), 0.1, 11, color='limegreen', alpha=0.2)
        wall3 = plt.Rectangle((0, 4.7), 5, 0.1, color='limegreen', alpha=0.2)

        # plot
        risky_width = self.obstacles[1][0] - self.obstacles[0][0] + 0.1
        risky_height = self.obstacles[1][1] - self.obstacles[0][1] - 0.1
        risky = plt.Rectangle((self.obstacles[0][0], self.obstacles[0][1]), risky_width, risky_height, color='r', alpha=0.5)

        # plot goal position
        goal_x = []
        goal_y = []
        for goal_r_I in self.goal_r_I:
            goal_x.append(goal_r_I[0])
            goal_y.append(goal_r_I[1])
        plt.plot(goal_x, goal_y, 'r*')

        ax.add_patch(wall1)
        ax.add_patch(wall2)
        ax.add_patch(wall3)
        ax.add_patch(risky)

        # plot waypoints
        # waypoints_x = []
        # waypoints_y = []
        # for waypoint in waypoints:
        #     waypoints_x.append(waypoint[0])
        #     waypoints_y.append(waypoint[1])
        # plt.plot(waypoints_x, waypoints_y, 'bo')

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
        car_body = [None] * self.uav_num
        c_x = [None] * self.uav_num
        c_y = [None] * self.uav_num
        q = [None] * self.uav_num

        car_height = 0.25
        car_width = 0.1


        line_traj_ref = [None] * self.uav_num
        for i in range(self.uav_num):
            line_traj[i], = ax.plot(position[i][:1, 0], position[i][:1, 1])
            c_x[i], c_y[i] = position[i][0, 0:2]
            q[i] = position[i][0, 2]


            car_body[i], = ax.plot(c_x[i], c_y[i], linewidth=1, color='red', marker='o', markersize=8)
            # car_body[i] = patches.Rectangle((c_x[i],c_y[i]), car_width, car_height, angle=np.rad2deg(q[i]))
            # ax.add_patch(car_body[i])


        # time label
        time_template = 'time = %.1fs'
        # time_text = ax.text(0.66, 0.55, "time", transform=ax.transAxes)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))
        sim_horizon = np.size(position[0], 0)

        def update_traj(num):
            # customize
            # time_text.set_text(time_template % (num * dt))

            # trajectory
            for i in range(self.uav_num):
                line_traj[i].set_data(position[i][:num, 0], position[i][:num, 1])
                c_x[i], c_y[i] = position[i][num, 0:2]
                q[i] = position[i][num, 2]
                car_body[i].set_data(c_x[i], c_y[i])

                # car_body[i].set_width(car_width)
                # car_body[i].set_height(car_height)
                # car_body[i].set_xy([c_x[i], c_y[i]])
                # car_body[i].set_angle(np.rad2deg(q[i]))


            return line_traj[0], car_body[0], line_traj[1], car_body[1], line_traj[2], car_body[2], #time_text,


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

    def obstacle_dis(self, obstacles, r_I):

            vec_obst = [obstacles[1][0] - obstacles[0][0], obstacles[1][1] - obstacles[0][1]]
            vec_r_I0 = [r_I[0] - obstacles[0][0], r_I[1] - obstacles[0][1]]
            vec_r_I1 = [r_I[0] - obstacles[1][0], r_I[1] - obstacles[1][1]]

            obstacle_d = []
            r_obst = (vec_obst[0] * vec_r_I0[0] + vec_obst[1] * vec_r_I0[1]) / (norm_2(vec_obst) ** 2)
            r2ob_dis = r_obst * vec_obst
            r2ob_dis = [vec_r_I0[0] - r2ob_dis[0], vec_r_I0[1] - r2ob_dis[1]]
            r2ob_dis = sqrt(r2ob_dis[0] ** 2 + r2ob_dis[1] ** 2)
            vec_r_I0 = sqrt(vec_r_I0[0] ** 2 + vec_r_I0[1] ** 2)
            vec_r_I1 = sqrt(vec_r_I1[0] ** 2 + vec_r_I1[1] ** 2)

            obstacle_d = vec_r_I1 * fmax(r_obst - 1, 0) / (r_obst - 1) + vec_r_I0 * fmin(r_obst, 0) / \
                         r_obst + r2ob_dis * fmin(r_obst - 1, 0) * fmax(r_obst, 0) / ((r_obst - 1) * r_obst)

            return obstacle_d


    def angle2rot(self,angle):
        R = vertcat(
            horzcat(math.cos(angle),math.sin(angle)),
            horzcat(-math.sin(angle), math.cos(angle))
        )
        return R



