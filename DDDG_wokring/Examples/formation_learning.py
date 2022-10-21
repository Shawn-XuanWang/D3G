from XEnv import XEnv
from Game_PDP import Game_PDP
from casadi import *
import scipy.io as sio
import numpy as np
import time

#import multiprocessing as mp

# --------------------------- define your fleet ---------------------------------------
num_uav = 4
uav_adj_matrix = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
dt = 0.05
# ---------------------------- Given parameters ----------------------------------------------
c_set = [0.01, 0.01, 0.01, 0.01]
wthrust_set = [0.005, 0.005, 0.005, 0.005]
base_parameter = [[ 1.2, 40, 4, 0, 0, 0, 0]] * 4
# [1 Jx, 2 Jy, 3 Jz, 4 mass, 5 w_F_v0, 6 w_r_formation0, 7 w_v_formation0, 8 uav_dist0, 9 w_uav_collision0]
# state_bd_set = [[-10, 10], [-10, 10], [-10, 10], [-10, 10]]

# --------------------------- load environment ----------------------------------------
uavswarm = [XEnv.UAV_formation() for _ in range(num_uav)]
for i in range(num_uav):
    uavswarm[i].initDyn(c=c_set[i], Jx=1, Jy=1,Jz=1,uav_index=i, dt=dt)
for i in range(num_uav):
    uavswarm[i].initCost(uavswarm=uavswarm, num_uav=num_uav, uav_index=i, w_F_r=1e2, wthrust=wthrust_set[i])

# --------------------------- load demos data ----------------------------------------
# To Be Filled
trajectories = [None] * 3
data = sio.loadmat('data/uav_formation_demos1.mat')
trajectories[0] = data['trajectories']
data = sio.loadmat('data/uav_formation_demos2.mat')
trajectories[1] = data['trajectories']
data = sio.loadmat('data/uav_formation_demos3.mat')
trajectories[2] = data['trajectories']
n_demo = 3
true_parameter = data['true_parameter']
dt = data['dt']

print('true parameter:')
print(true_parameter)
true_parameter = [true_parameter[i][3:] for i in range(num_uav)]
print(true_parameter)
# print('base parameter:')
# print(base_parameter)

# ------------------------- Define Game --------------------------
uav_game = Game_PDP.DGSys()
uav_game.setGraph(num_robot=num_uav, adj_matrix=uav_adj_matrix)
uav_game.setVariables(uavswarm)
uav_game.setDyn(uavswarm)
uav_game.setCosts(uavswarm)
uav_game.def_DPMP()


# ---------------Computing-------------------
#pool = mp.Pool(min(n_demo,mp.cpu_count()))
# ------------------------- Learn dynamics & Objectives --------------------------

for tr in range(1):  # trial loop
    start_time = time.time()
    lr = 1e-4  # learning rate
    # initialize
    loss_trace = []
    parameter_trace = [[]] * num_uav
    sigma = 0.5
    initial_parameter = [None] * num_uav
    current_parameter = [None] * num_uav
    for i in range(num_uav):
        initial_parameter[i] = true_parameter[i] + sigma * true_parameter[i] * np.random.random(len(true_parameter[i])) - sigma * true_parameter[i] / 2
        current_parameter[i] = initial_parameter[i]

    print(initial_parameter)
    curr_sys_traj = [None] * n_demo
    for k in range(int(1e2)):  # iteration loop (or epoch loop)
        loss = 0
        # Gradient of parameter
        dp = [np.zeros(current_parameter[i].shape)] * num_uav
        # loop for each demos trajectory
        for d in range(n_demo):

            demo_state_traj = [None] * num_uav
            demo_control_traj = [None] * num_uav
            demo_ini_state = [None] * num_uav
            for i in range(num_uav):
                demo_state_traj[i] = trajectories[d]['state_traj_opt'][0, 0][i]
                demo_control_traj[i] = trajectories[d]['control_traj_opt'][0, 0][i]
                demo_ini_state[i] = demo_state_traj[i][0, :]
            demo_horizon = demo_state_traj[0].shape[0]

            # Solve the dynamic game
            if k == 0:
                curr_sys_traj[d] = uav_game.dyn_game_Solver(ini_state=demo_ini_state, horizon=demo_horizon,
                                                            auxvar_value=base_parameter,
                                                            gamma=1e-4, print_gap=1e3,
                                                            init_option=1, eph=1e3, dt=dt)

                curr_sys_traj[d] = uav_game.dyn_game_Solver(ini_state=demo_ini_state, horizon=demo_horizon,
                                                            auxvar_value=current_parameter,
                                                            loaded_traj=curr_sys_traj[d],
                                                            gamma=1e-4, print_gap=1e3,
                                                            init_option=0, eph=1e3, dt=dt)
            else:
                curr_sys_traj[d] = uav_game.dyn_game_Solver(ini_state=demo_ini_state, horizon=demo_horizon,
                                                            auxvar_value=current_parameter,
                                                            loaded_traj=curr_sys_traj[d],
                                                            gamma=1e-4, print_gap=1e3,
                                                            init_option=0, eph=1e3, dt=dt)

            # Solve dldxi
            dldx_traj = [None] * num_uav
            dldu_traj = [None] * num_uav

            for i in range(num_uav):
                dldx_traj[i] = curr_sys_traj[d]['state_traj_opt'][i] - demo_state_traj[i]
                dldu_traj[i] = curr_sys_traj[d]['control_traj_opt'][i] - demo_control_traj[i]

            # Solve DPMP
            [dxdTheta, dudTheta] = uav_game.D_PMP_Solver(horizon=demo_horizon, auxvar_value=current_parameter,
                                                         loaded_traj=curr_sys_traj[d])

            for i in range(num_uav):
                loss = loss + numpy.linalg.norm(dldx_traj[i]) ** 2 + numpy.linalg.norm(dldu_traj[i]) ** 2

                for t in range(demo_horizon - 1):
                    dxdtheta_current = np.hsplit(dxdTheta[t][i], num_uav)
                    dudtheta_current = np.hsplit(dudTheta[t][i], num_uav)

                    dp[i] = dp[i] + np.matmul(dldx_traj[i][t, :], dxdtheta_current[i]) + np.matmul(dldu_traj[i][t, :],
                                                                                                   dudtheta_current[i])
                dxdtheta_end = np.hsplit(dxdTheta[-1][i], num_uav)
                dp[i] = dp[i] + numpy.dot(dldx_traj[i][-1, :], dxdtheta_end[i])
        # end loop demos

        # take the expectation (average)
        for i in range(num_uav):
            dp[i] = dp[i] / n_demo
            # update
            current_parameter[i] = current_parameter[i] - lr * dp[i]
            parameter_trace[i] += [current_parameter[i]]
        loss = loss / n_demo
        loss_trace += [loss]
        print('current_parameter')
        for i in range(num_uav):
            print(current_parameter[i])
        # print(np.shape(dp))
        # print(loss)

        # print and terminal check
        if k % 1 == 0:
            print('trial:', tr, 'iter:', k, ' loss: ', loss_trace[-1].tolist())

        sio.savemat('data/uav_formation_learning.mat', {'parameter_trace': parameter_trace, 'loss_trace': loss_trace,
                                                      'true_parameter': true_parameter})
# save
#def compute_gradient(d, trajectories, curr_sys_traj, )
