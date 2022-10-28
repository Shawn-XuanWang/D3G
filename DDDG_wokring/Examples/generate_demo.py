from casadi import *
from XEnv import GEnv as XEn
from Game_PDP import Game_GPDP as GPDP
from casadi import *
import scipy.io as sio
import numpy as np
import time
import pickle

# Set initial state and parameters
num_uav = 4    # number of UAVs is fixed, do not change
#ini_state = []
ini_r_I = [[2, 0.25], [1, 0.1], [4, 0.15], [3, 0.08]]
ini_q = [0.5*pi]
#[ini_state.append(ini_r_I[i] + ini_v_I + ini_q + ini_w) for i in range(num_uav)]
ini_state = [(ini_r_I[0] + ini_q),
            (ini_r_I[1] + ini_q),
            (ini_r_I[2] + ini_q),
            (ini_r_I[3] + ini_q)]



# Initial the environment
uav_adj_matrix = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
dt = 0.05
UAV_team = XEn.UAV_formation(uav_num=num_uav)
UAV_team.initDyn(dt=dt)
UAV_team.initCost()

# define the game solver
uav_game = GPDP.DGSys()
uav_game.setGraph(num_robot=num_uav, adj_matrix=uav_adj_matrix)
uav_game.setVariables(UAV_team)
uav_game.setDyn(UAV_team)
uav_game.setCosts(UAV_team)
uav_game.def_DPMP()

restart = 1
formation_horizon = 100
true_parameter = [[50, 0.1, 1, 1, 1]] * num_uav
base_parameter = [[100, 0.1, 1, 1, 10]] * num_uav

if restart == 1:
    traj = uav_game.dyn_game_Solver(ini_state=ini_state, horizon=formation_horizon, auxvar_value=base_parameter,
                             init_option=1, eph=1e3, gamma=1e-4, solver_option='d', dt=dt)
    sio.savemat('data/uav_formation_demos1.mat', {'trajectories': traj,
                                       'dt': dt,
                                       'true_parameter': true_parameter})
else:
    data = sio.loadmat('data/uav_formation_demos1.mat')
    load_traj = data['trajectories'][0][0]
    traj = uav_game.dyn_game_Solver(ini_state=ini_state, horizon=formation_horizon, auxvar_value=true_parameter, print_level=1,
                                init_option=0, eph = 1e4, print_gap=1e3, gamma=1e-5, solver_option='cs', loaded_traj=load_traj, dt=dt)

    sio.savemat('data/uav_formation_demos1.mat', {'trajectories': traj,
                                  'dt': dt,
                                   'true_parameter': true_parameter})
UAV_team.play_multi_animation(dt=dt, state_traj=traj['state_traj_opt'],save_option=0)