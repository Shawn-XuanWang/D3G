from XEnv import XEnv
from Game_PDP import Game_PDP
from casadi import *
import scipy.io as sio
import numpy as np
import time

# import multiprocessing as mp

# --------------------------- define your fleet ---------------------------------------
#num_uav = 3
#uav_adj_matrix = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
num_uav = 4
uav_adj_matrix = [[1, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1,0,1,1]]
dt = 0.1
# ---------------------------- Given parameters ----------------------------------------------
wthrust_set = [0.1, 0.1, 0.1, 0.1]
# [ 4 mass, 5 w_F_v0, 6 w_r_formation0, 7 w_v_formation0, 8 uav_dist0, 9 w_uav_collision0]
# [1 Jx, 4 mass, 5 w_F_v0, 6 w_r_formation0, 7 w_v_formation0, 8 uav_dist0, 9 w_uav_collision0]
# state_bd_set = [[-10, 10], [-10, 10], [-10, 10], [-10, 10]]

# --------------------------- load demos data ----------------------------------------
# To Be Filled
n_demo = 3
trajectories = [None] * n_demo
data = sio.loadmat('data/uav_formation_demos1.mat')
trajectories[0] = data['trajectories']
data = sio.loadmat('data/uav_formation_demos2.mat')
trajectories[1] = data['trajectories']
data = sio.loadmat('data/uav_formation_demos3.mat')
trajectories[2] = data['trajectories']
#data = sio.loadmat('data/uav_formation_demos4.mat')
#trajectories[3] = data['trajectories']


true_parameter = data['true_parameter']
dt = data['dt']

print('true parameter:')
print(true_parameter)
true_parameter = [np.array([true_parameter[0][j] for j in [1, 5, 7]])] * num_uav
print('New true parameter:')
print(true_parameter)
# print('base parameter:')
# print(base_parameter)

# --------------------------- load environment ----------------------------------------
uavswarm = [XEnv.UAV_formation_2D() for _ in range(num_uav)]
for i in range(num_uav):
    uavswarm[i].initDyn(Jxy=1, uav_index=i, dt=dt)
for i in range(num_uav):
    uavswarm[i].initCost(adj_matrix=uav_adj_matrix[i], uavswarm=uavswarm, num_uav=num_uav, uav_index=i, w_F_r=1e2,
                         w_F_v=50, w_r_formation=0.2, w_v_formation=0.4, w_uav_collision=10, wthrust=wthrust_set[i])
# Parameters to learn: mass, uav_dist, w_obst

# ------------------------- Define Game --------------------------
uav_game = Game_PDP.DGSys()
uav_game.setGraph(num_robot=num_uav, adj_matrix=uav_adj_matrix)
uav_game.setVariables(uavswarm)
uav_game.setDyn(uavswarm)
uav_game.setCosts(uavswarm)
uav_game.def_DPMP()

# ---------------Computing-------------------
# pool = mp.Pool(min(n_demo,mp.cpu_count()))
# ------------------------- Learn dynamics & Objectives --------------------------

for tr in range(1):  # trial loop
    start_time = time.time()
    lr = 1e-2  # learning rate
    gain = 1
    # initialize
    loss_trace = []
    dp_trace = []
    dp_norm_trace=[]
    parameter_trace = []
    dldx_traj_trace = []
    dldu_traj_trace = []
    dxdTheta_trace = []
    dudTheta_trace = []
    sigma = 0.5
    initial_parameter = [None] * num_uav
    current_parameter = [None] * num_uav
    for i in range(num_uav):
        initial_parameter[i] = true_parameter[i] + sigma *  np.random.random(
            len(true_parameter[i])) - sigma  / 2
        current_parameter[i] = initial_parameter[i]

    print('Initial_parameter:')
    print(initial_parameter)
    curr_sys_traj = [None] * n_demo
    for k in range(int(1e4)):  # iteration loop (or epoch loop)
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
                                                            auxvar_value=current_parameter,
                                                            gamma=1e-4, print_gap=1e2,
                                                            init_option=1, eph=2e3, dt=dt)

                curr_sys_traj[d] = uav_game.dyn_game_Solver(ini_state=demo_ini_state, horizon=demo_horizon,
                                                            auxvar_value=current_parameter,
                                                            loaded_traj=curr_sys_traj[d],
                                                            gamma=1e-4, print_gap=1e3, solver_option='cs',
                                                            init_option=0, eph=3e3, dt=dt)

                print('iter:', k, 'demo ', d, ' solved!')
            else:
                curr_sys_traj[d] = uav_game.dyn_game_Solver(ini_state=demo_ini_state, horizon=demo_horizon,
                                                            auxvar_value=current_parameter,
                                                            loaded_traj=curr_sys_traj[d],
                                                            gamma=1e-4, print_gap=1e3, solver_option='d',
                                                            init_option=0, eph=5e2, dt=dt)

                flag=0
                while True or flag<100:
                    flag = flag + 1
                    try:
                        curr_sys_traj[d] = uav_game.dyn_game_Solver(ini_state=demo_ini_state, horizon=demo_horizon,
                                                            auxvar_value=current_parameter,
                                                            loaded_traj=curr_sys_traj[d],
                                                            gamma=1e-4, print_gap=1e3, solver_option='cs',
                                                            init_option=0, eph=4e3, dt=dt)
                    except:
                        continue

                    else:
                        # the rest of the code
                        break
                    print('Another try', flag)

                print('iter:', k, 'demo ', d, ' solved!')
            # Solve dldxi
            dldx_traj = [None] * num_uav
            dldu_traj = [None] * num_uav

            for i in range(num_uav):
                dldx_traj[i] = curr_sys_traj[d]['state_traj_opt'][i] - demo_state_traj[i]
                dldu_traj[i] = curr_sys_traj[d]['control_traj_opt'][i] - demo_control_traj[i]

            dldx_traj_trace += [dldx_traj[:]]
            dldu_traj_trace += [dldu_traj[:]]
            # Solve DPMP
            [dxdTheta, dudTheta] = uav_game.D_PMP_Solver(horizon=demo_horizon, auxvar_value=current_parameter,
                                                         loaded_traj=curr_sys_traj[d])
            dxdTheta_trace += [dxdTheta[:]]
            dudTheta_trace += [dudTheta[:]]
            for i in range(num_uav):
                loss = loss + numpy.linalg.norm(dldx_traj[i]) ** 2 + numpy.linalg.norm(dldu_traj[i]) ** 2

                for t in range(demo_horizon - 1):
                    dxdtheta_current = np.hsplit(dxdTheta[t][i], num_uav)
                    dudtheta_current = np.hsplit(dudTheta[t][i], num_uav)
                    for j in uav_game.neighbors[i]:    # range(num_uav)
                        dp[i] = dp[i] + np.matmul(dldx_traj[j][t, :], dxdtheta_current[j]) + np.matmul(
                            dldu_traj[j][t, :], dudtheta_current[j])

                dxdtheta_end = np.hsplit(dxdTheta[-1][i], num_uav)
                for j in uav_game.neighbors[i]:
                    dp[i] = dp[i] + numpy.dot(dldx_traj[j][-1, :], dxdtheta_end[j])
        # end loop demos

        # take the expectation (average)
        for i in range(num_uav):
            # dp[i] = dp[i] / n_demo / num_uav
            dp[i] = dp[i]/ numpy.linalg.norm(dp[i])
            # update
            current_parameter[i] = current_parameter[i] - lr * gain * dp[i]

        gain = 2000/(k+2000)
        dp_trace += [dp[:]]
        dp_norm_trace += [dp[:]]
        parameter_trace += [current_parameter[:]]

        loss = loss / n_demo
        loss_trace += [loss]
        # print(np.shape(dp))
        # print(loss)

        # print and terminal check
        if k % 1 == 0:
            print('trial:', tr, 'iter:', k, ' loss: ', loss_trace[-1].tolist())

        sio.savemat('data/uav_formation_learning.mat',
                    {'parameter_trace': parameter_trace, 'loss_trace': loss_trace, 'dxdTheta_trace': dxdTheta_trace,
                     'dp_trace': dp_trace, 'dldu_traj_trace': dldu_traj_trace, 'dldx_traj_trace': dldx_traj_trace,
                     'dudTheta_trace': dudTheta_trace, 'true_parameter': true_parameter})
# save
# def compute_gradient(d, trajectories, curr_sys_traj, )
