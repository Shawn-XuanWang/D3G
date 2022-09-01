from XEnv import XEnv
from Game_PDP import Game_PDP
from casadi import *
import scipy.io as sio
import numpy as np

# --------------------------- define your fleet ---------------------------------------
# num_uav = 3
# uav_adj_matrix = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
num_uav = 4
uav_adj_matrix = [[1, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]]
dt = 0.1
# ---------------------------- Given parameters ----------------------------------------------
wthrust_set = [0.1, 0.1, 0.1, 0.1]
# w_F_r=1e2; w_F_v=50; w_r_formation=0.2; w_v_formation=0.4; w_uav_collision=10

# --------------------------- load demos data ----------------------------------------
# To Be Filled
data = sio.loadmat('data/uav_formation_demos2.mat')
trajectories = data['trajectories']

true_parameter = data['true_parameter']
dt = data['dt']

print('Full parameter:')
print(true_parameter)
true_parameter = [np.array([true_parameter[0][j] for j in [1, 5, 7]])] * num_uav
print('Parameters to learn:')
print(true_parameter)

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


# ------------------------ Initial Game solver------------------------
demo_state_traj = [None] * num_uav
demo_control_traj = [None] * num_uav
demo_ini_state = [None] * num_uav
for i in range(num_uav):
    demo_state_traj[i] = trajectories['state_traj_opt'][0, 0][i]
    demo_control_traj[i] = trajectories['control_traj_opt'][0, 0][i]
    demo_ini_state[i] = demo_state_traj[i][0, :]
demo_horizon = demo_state_traj[0].shape[0]

opttraj = {"state_traj_opt": demo_state_traj,
                           "control_traj_opt": demo_control_traj}

loss_trace = []

# ------------------------ Solve the Game ------------------------
for k in range(int(1e7)):

    if k == 0:
        curr_sys_traj = uav_game.dyn_game_Solver(ini_state=demo_ini_state, horizon=demo_horizon,
                                                 auxvar_value=true_parameter,
                                                 gamma=1e-4, print_gap=1e2,
                                                 init_option=1, eph=1e1, dt=dt)
    else:
        curr_sys_traj = uav_game.dyn_game_Solver(ini_state=demo_ini_state, horizon=demo_horizon,
                                                 auxvar_value=true_parameter,
                                                 loaded_traj=curr_sys_traj,
                                                 gamma=1e-3*(1e5/(1e5+k)), print_gap=1e3, solver_option='d',
                                                 init_option=0, eph=1e1, dt=dt)
    loss = 0
    for i in range(num_uav):
        loss += numpy.linalg.norm(
            curr_sys_traj['state_traj_opt'][i] - demo_state_traj[i]) + numpy.linalg.norm(
            curr_sys_traj['control_traj_opt'][i]- demo_control_traj[i])
    loss_trace += [loss]
    print('k',k, 'loss:', loss)

    if k % 100 ==0:
        sio.savemat('data/test_game_solver.mat', {'loss_trace': loss_trace})

