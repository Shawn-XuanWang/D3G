from casadi import *
from XEnv import EnvUAV as XEn
from Game_PDP import Game_Parallel as GPDP
import multiprocessing as mp
from casadi import *
import scipy.io as sio
import numpy as np
import time
import pickle

def Rob_Env_init():
    #define global variable
    global num_uav, ini_state, eph

    # Set initial state and parameters
    num_uav = 4  # number of UAVs is fixed, do not change
    # ini_state = []
    ini_r_I = [[2, 0.25, 4], [1, 0.1, 5.3], [4, 0.15, 5], [3, 0.08, 5.5]]
    # ini_r_I = np.ceil([10*np.random.randn(3) for _ in range(num_uav)]).tolist()
    ini_v_I = [0.0, 0.0, 0.0]
    ini_q = XEn.toQuaternion(0, [1, -1, 1])
    ini_w = [0.0, 0.0, 0.0]
    # [ini_state.append(ini_r_I[i] + ini_v_I + ini_q + ini_w) for i in range(num_uav)]
    ini_state = [(ini_r_I[0] + ini_v_I + ini_q + ini_w),
                 (ini_r_I[1] + ini_v_I + ini_q + ini_w),
                 (ini_r_I[2] + ini_v_I + ini_q + ini_w),
                 (ini_r_I[3] + ini_v_I + ini_q + ini_w)]

    # Set the parameters
    eph = 10
    c = 0.01
    wthrust = 0.005
    w_F_r = 0.01
    w_r_formation = 0.5

    # Initial the environment
    dt = 0.05
    UAV_team = XEn.UAV_formation(uav_num=num_uav)
    UAV_team.initDyn(c=c, dt=dt)
    UAV_team.initCost(w_F_r=w_F_r, wthrust=wthrust)  # w_F_r is terminal cost

    return UAV_team


def Rob_Game_init(UAV_team, robot_index, num_uav, ini_state):

    global horizon, true_parameter, ini_ctrl_traj
    # create the communication graph
    uav_adj_matrix = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

    # define GamePDP
    uav_game = GPDP.DGSys(robot_index)
    uav_game.setGraph(num_robot=num_uav, adj_matrix=uav_adj_matrix)
    uav_game.setVariables(UAV_team)
    uav_game.setDyn()
    uav_game.setCosts(UAV_team)
    uav_game.def_DPMP()

    # initialization
    horizon = 80
    true_parameter = [1, 1, 1, 1, 50, 5, 5, 1, 20, 50]
    ini_ctrl_traj = np.random.rand(horizon - 1, uav_game.n_ctrl_var)

    return uav_game




if __name__ =="__main__":

    # init environment
    UAV_team = Rob_Env_init()


    # init game_solver
    uav_game = [None] * num_uav
    for robot_index in range(num_uav):
        uav_game[robot_index] = Rob_Game_init(UAV_team, robot_index, num_uav, ini_state)


    # init the trajectory
    state_traj = ini_state
    ctrl_traj = ini_ctrl_traj

    # solve the game
    for tau in range(eph):
        for robot_index in range(num_uav):
            # compute state traj
            uav_game[robot_index].Compute_state_traj(state_traj=state_traj[robot_index], ctrl_traj=ctrl_traj,
                                                     horizon=horizon, auxvar_value=true_parameter)

        # get neighbors' state
        current_state_vec = []
        for robot_index in uav_game[robot_index].neighbors:
            current_state_vec += [uav_game[robot_index].tau_state_traj]

        # print(len(current_state_vec))

        for robot_index in uav_game[robot_index].neighbors:
            uav_game[robot_index].Get_neighbor_state(current_state_vec)

        # compute co-state traj
        for robot_index in range(num_uav):
            uav_game[robot_index].Compute_Costate_traj(loaded_traj=[None], gamma=1e-4, print_level=0,
                                                       init_option=0, print_gap=1e2, solver_option='d')

