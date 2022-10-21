from casadi import *
from XEnv import EnvUAV as XEn
from Game_PDP import Game_PDP_new as GPDP
from casadi import *
import scipy.io as sio
import numpy as np
import time
import pickle

# Set initial state and parameters
num_uav = 4    # number of UAVs is fixed, do not change
#ini_state = []
ini_r_I = [[2, 0.25, 4], [1, 0.1, 5.3], [4, 0.15, 5], [3, 0.08, 5.5]]
#ini_r_I = np.ceil([10*np.random.randn(3) for _ in range(num_uav)]).tolist()
ini_v_I = [0.0, 0.0, 0.0]
ini_q = XEn.toQuaternion(0, [1, -1, 1])
ini_w = [0.0, 0.0, 0.0]
#[ini_state.append(ini_r_I[i] + ini_v_I + ini_q + ini_w) for i in range(num_uav)]
ini_state = [(ini_r_I[0] + ini_v_I + ini_q + ini_w),
            (ini_r_I[1] + ini_v_I + ini_q + ini_w),
            (ini_r_I[2] + ini_v_I + ini_q + ini_w),
            (ini_r_I[3] + ini_v_I + ini_q + ini_w)]
c = 0.01

# Set cost parameters
wthrust = 0.005
w_F_r = 0.01
w_r_formation = 0.5

# Initial the environment
uav_adj_matrix = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
dt = 0.05
UAV_team = XEn.UAV_formation(uav_num=num_uav)
UAV_team.initDyn(c=c,dt=dt)
UAV_team.initCost(w_F_r=w_F_r, wthrust=wthrust) # w_F_r is terminal cost

# --------------------------- define GamePDP ---------------------------------------------
uav_game = GPDP.DGSys()
uav_game.setGraph(num_robot=num_uav, adj_matrix=uav_adj_matrix)
uav_game.setVariables(UAV_team)
uav_game.setDyn(UAV_team)
uav_game.setCosts(UAV_team)
uav_game.def_DPMP()

# -------------------------- solve Game ------------------------------------
restart = 1
formation_horizon = 80
true_parameter = [[1, 1, 1, 1, 50, 5, 5, 1, 20, 50]] * num_uav
base_parameter = [[1, 1, 1, 1, 50, 5, 5, 1, 0, 0]] * num_uav

if restart == 1:
    traj = uav_game.dyn_game_Solver(ini_state=ini_state, horizon=formation_horizon, auxvar_value=base_parameter,
                             init_option=1, eph=1e3, gamma=1e-4, dt=dt)
    sio.savemat('data_new/uav_formation_demos1.mat', {'trajectories': traj,
                                       'dt': dt,
                                       'true_parameter': true_parameter})
else:
    data = sio.loadmat('data_new/uav_formation_demos1.mat')
    load_traj = data['trajectories'][0][0]
    traj = uav_game.dyn_game_Solver(ini_state=ini_state, horizon=formation_horizon, auxvar_value=true_parameter, print_level=1,
                                init_option=0, eph = 1e4, print_gap=1e3, gamma=1e-5, solver_option='cs', loaded_traj=load_traj, dt=dt)

    sio.savemat('data/uav_formation_demos2.mat', {'trajectories': traj,
                                  'dt': dt,
                                   'true_parameter': true_parameter})
UAV_team.play_multi_animation(wing_len=0.2, dt=dt, state_traj=traj['state_traj_opt'],save_option=0)