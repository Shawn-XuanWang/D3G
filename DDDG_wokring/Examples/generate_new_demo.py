from XEnv import EnvUAV as XEn
from Game_PDP import Game_PDP_waypoint as GPDP
from casadi import *
import scipy.io as sio
import numpy as np
import time

# --------------------------- define your fleet ---------------------------------------
uav_adj_matrix = [[1, 1], [1, 1]]
num_uav = 2
obstacles = [[2.5, 1], [2.5, 3]]
wall = [0, 5]

# ----------------------------parameters ----------------------------------------------
wthrust_set = [0.1, 0.1]
# state_bd_set = [[-10, 10], [-10, 10], [-10, 10], [-10, 10]]

#demo 1
#ini_r_I = [[3, 0.25], [1, 0.1], [2.2, 0], [4.1, 0.05]]
#demo 2
#ini_r_I = [[2.1, 0.25], [0.8, 0.2], [4, 0.1] ,[3.1, 0]]
#demo 3
ini_r_I = [[0.4, 0.25], [0.8, 0.1]]
#demo 4
#ini_r_I = [[1, 0.25], [2.2, 0.1], [4, 0.15], []]
ini_v_I = [0.0, 0.0]
ini_theta = [0.0]
ini_w = [0.0]
uav_ini_state = [(ini_r_I[0] + ini_v_I + ini_theta + ini_w),
                 (ini_r_I[1] + ini_v_I + ini_theta + ini_w)]
goal_r_I = [[3.5, 1.0], [4.0, 0.6], [4.6, 0.6], [3.8,1]]
formation_horizon = 50
waypoints = [[1.0, 1.0], [2.5, 2.5], [3, 3.5], [3.5, 4], [3.8, 3.5], goal_r_I[0]]

# true_parameter = [[1, 1, 1, 1, 0.3,  5, 5, 10, 1, 1]] * 4
#                  1  2  3  4  5  6  7  8  9
true_parameter = [[1, 1, 50, 0.2, 0.4, 1, 10, 10, 10]] * num_uav
base_parameter = [[1, 1, 50, 10, 0.4, 0.4, 10, 10, 100]] * num_uav

restart = 1
# [1 Jx, 2 mass, 3 w_F_v0, 4 w_r_formation0, 5 w_v_formation0, 6 uav_dist0, 7 w_uav_collision0, 8 w_obst, 9 w_waypoint]

# Initial the environment
dt = 0.05
UAV_team = XEn.UAV_formation_2D(uav_num=num_uav, obstacles=obstacles, wall=wall)
UAV_team.initDyn(dt=dt)
UAV_team.initCost(adj_matrix=uav_adj_matrix, w_F_r=1, wthrust=wthrust_set, goal_r_I=goal_r_I) # w_F_r is terminal cost

# --------------------------- define GamePDP ---------------------------------------------
uav_game = GPDP.DGSys()
uav_game.setGraph(num_robot=num_uav, adj_matrix=uav_adj_matrix)
uav_game.setVariables(UAV_team)
uav_game.setDyn(UAV_team)
uav_game.setCosts(UAV_team)
uav_game.def_DPMP()

# -------------------------- solve Game ------------------------------------
if restart == 1:
    traj = uav_game.dyn_game_Solver(ini_state=uav_ini_state, horizon=formation_horizon, auxvar_value=base_parameter,
                             init_option=1, eph=2e3, gamma=1e-4, dt=dt, waypoints=waypoints)
    sio.savemat('data/uav_formation_demos3.mat', {'trajectories': traj,
                                       'dt': dt,
                                       'true_parameter': true_parameter})
else:
    data = sio.loadmat('data/uav_formation_demos3.mat')
    load_traj = data['trajectories'][0][0]
    traj = uav_game.dyn_game_Solver(ini_state=uav_ini_state, horizon=formation_horizon, auxvar_value=true_parameter,
                                init_option=0, eph = 5e3, print_gap=1e3, gamma=1e-4, solver_option='cs', loaded_traj=load_traj, dt=dt)

    sio.savemat('data/uav_formation_demos3.mat', {'trajectories': traj,
                                  'dt': dt,
                                   'true_parameter': true_parameter})
UAV_team.play_multi_animation_3(wing_len=0.2, dt=dt, state_traj=traj['state_traj_opt'],save_option=0)