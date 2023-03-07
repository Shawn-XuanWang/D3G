from casadi import *
from GEnv import GEnv as XEn
from Game_PDP import Game_GPDP as GPDP
from casadi import *
import scipy.io as sio
import numpy as np
import time
import pickle

# Set initial state and parameters
num_uav = 3    # number of UAVs is fixed, do not change
#ini_state = []
# demo 1
ini_r_I = [[0.30, 0.4], [0.4, 1.4], [0.4, 2.4]]

# demo 2
# ini_r_I = [[0.3, 0.1], [0.3, 0.8], [0.4, 1.5]]

goal_r_I = [[4.4, 0.5],[4.4, 1.5],[4.4, 2.5]]
# goal_r_I = [[3.8, 1.6], [4.1, 1.5], [4.4, 1.5]]
ini_q = [0.5*pi]
inv_v = [0.00, 0.00]
waypoints = [ini_r_I[1], [1.5, 3.4], [2.5, 4.6], [3.5, 3.2], goal_r_I[1]]

#[ini_state.append(ini_r_I[i] + ini_v_I + ini_q + ini_w) for i in range(num_uav)]
ini_state = [(ini_r_I[0] + ini_q + inv_v),
            (ini_r_I[1] + ini_q + inv_v),
            (ini_r_I[2] + ini_q + inv_v)]


# Initial the environment
uav_adj_matrix = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
obstacles = [[2.5, 0.5], [2.5, 3.3]]
wall_vet = [0, 5]
wall_hor = [4.6]
dt = 0.05
UAV_team = XEn.UAV_formation(uav_num=num_uav, obstacles=obstacles, wall_hor=wall_hor,wall_vet=wall_vet,goal_r_I=goal_r_I)
UAV_team.initDyn(dt=dt)
UAV_team.initCost(adj_matrix=uav_adj_matrix)

# define the game solver
uav_game = GPDP.DGSys()
uav_game.setGraph(num_robot=num_uav, adj_matrix=uav_adj_matrix)
uav_game.setVariables(UAV_team)
uav_game.setDyn(UAV_team)
uav_game.setCosts(UAV_team)
uav_game.def_DPMP()

restart = 1
formation_horizon = 50

true_parameter = [[1, 1, 1, 1.5, 1.0, 0.6, 0.0, 12],
                  [1, 1, 1, 1.5, 1.0, 0.6, 0.1, 12],
                  [1, 1, 1, 1.5, 1.0, 0.6, 0.2, 12]] # parameter: w_F_r, w_F_v, w_r_formation, w_v_formation, uav_dist, w_uav_collision, w_obst, w_waypoint
base_parameter = [[10, 1, 0.00, 1.5, 0.9, 1.0, 0.0, 5],
                  [10, 1, 0.00, 1.5, 0.9, 1.0, 0.01, 10],
                  [10, 1, 0.00, 1.5, 0.9, 1.0, 0.01, 5]]

if restart == 1:
    traj = uav_game.dyn_game_Solver(ini_state=ini_state, horizon=formation_horizon, auxvar_value=base_parameter,
                             init_option=1, eph=1e3, gamma=1.2e-3, solver_option='d', dt=dt, waypoints=waypoints)
    sio.savemat('data/uav_formation_demos_init3.mat', {'trajectories': traj,
                                       'dt': dt,
                                       'true_parameter': true_parameter})
else:
    data = sio.loadmat('data/uav_formation_demos_init3.mat')
    load_traj = data['trajectories'][0][0]
    traj = uav_game.dyn_game_Solver_new(ini_state=ini_state, horizon=formation_horizon, auxvar_value=true_parameter, print_level=1,
                                init_option=0, eph=1e4, print_gap=1e2, gamma=1.2e-3, solver_option='cs', waypoints=waypoints, loaded_traj=load_traj, dt=dt)

    sio.savemat('data/uav_formation_demos3.mat', {'trajectories': traj,
                                  'dt': dt,
                                   'true_parameter': true_parameter})
UAV_team.play_multi_animation(dt=dt, state_traj=traj['state_traj_opt'],save_option=0, waypoints=waypoints)