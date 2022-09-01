from XEnv import XEnv
from Game_PDP import Game_PDP
from casadi import *
import scipy.io as sio
import numpy as np
import time

# --------------------------- define your fleet ---------------------------------------
num_uav = 4
uav_adj_matrix = [[1, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1,0,1,1]]
dt = 0.1
# ----------------------------parameters ----------------------------------------------
wthrust_set = [0.1, 0.1, 0.1, 0.1]
# state_bd_set = [[-10, 10], [-10, 10], [-10, 10], [-10, 10]]

#demo 1
#ini_r_I = [[3, 0.25], [1, 0.1], [2.2, 0], [4.1, 0.05]]
#demo 2
#ini_r_I = [[2.1, 0.25], [0.8, 0.2], [4, 0.1] ,[3.1, 0]]
#demo 3
ini_r_I = [[1, 0.25], [3, 0.1], [2, 0.15], [4.3, 0]]
#demo 4
#ini_r_I = [[1, 0.25], [2.2, 0.1], [4, 0.15], []]
ini_v_I = [0.0, 0.0]
ini_theta = [0.0]
ini_w = [0.0]
uav_ini_state = [(ini_r_I[0] + ini_v_I + ini_theta + ini_w),
                 (ini_r_I[1] + ini_v_I + ini_theta + ini_w),
                 (ini_r_I[2] + ini_v_I + ini_theta + ini_w),
                 (ini_r_I[3] + ini_v_I + ini_theta + ini_w)]
formation_horizon = 30
# true_parameter = [[1, 1, 1, 1, 0.3,  5, 5, 10, 1, 1]] * 4
#                  1  2  3  4  5  6  7  8  9
true_parameter = [[1, 1, 50, 0.2, 0.4, 1, 10, 10]] *num_uav
base_parameter = [[1, 1, 50, 0.2, 0.4, 1, 10, 10]] *num_uav

restart = 0
# [1 Jx, 4 mass, 5 w_F_v0, 6 w_r_formation0, 7 w_v_formation0, 8 uav_dist0, 9 w_uav_collision0]
# --------------------------- load environment ----------------------------------------
uavswarm = [XEnv.UAV_formation_2D() for _ in range(num_uav)]
for i in range(num_uav):
    uavswarm[i].initDyn(uav_index=i, dt=dt)
for i in range(num_uav):
    uavswarm[i].initCost(adj_matrix=uav_adj_matrix[i], uavswarm=uavswarm, num_uav=num_uav, uav_index=i, w_F_r=1e2, wthrust=wthrust_set[i])

# --------------------------- load demos data ----------------------------------------
# To Be Filled

# --------------------------- define GamePDP ---------------------------------------------
uav_game = Game_PDP.DGSys()
uav_game.setGraph(num_robot=num_uav, adj_matrix=uav_adj_matrix)
uav_game.setVariables(uavswarm)
uav_game.setDyn(uavswarm)
uav_game.setCosts(uavswarm)
uav_game.def_DPMP()

# -------------------------- solve Game ------------------------------------
if restart == 1:
    traj = uav_game.dyn_game_Solver(ini_state=uav_ini_state, horizon=formation_horizon, auxvar_value=base_parameter,
                             init_option=1, eph=2e3, gamma=1e-4, dt=dt)
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
uavswarm[0].play_multi_animation_3(uavswarm=uavswarm, wing_len=0.2, dt=dt, state_traj=traj['state_traj_opt'],save_option=0)

# save

