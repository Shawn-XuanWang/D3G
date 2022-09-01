import sys
from casadi import *
import numpy as np
import scipy as spy
import scipy.io as sio


class DGSys:
    def __init__(self, project_name="my dynamic game system"):
        self.project_name = project_name
        self.graph = None
        self.num_uav = None
        self.full_aux = None

    def setGraph(self, num_robot=None, adj_matrix=None):
        if num_robot is None:
            sys.exit('Number of robots needs to be defined!')

        if adj_matrix is None or np.asarray(adj_matrix).shape != (num_robot, num_robot):
            sys.exit('Adjacency matrix needs to be properly defined!')
        self.graph = adj_matrix

        self.num_uav = num_robot

        self.neighbors = [None] * self.num_uav
        for i in range(self.num_uav):
            self.neighbors[i] = np.where(np.array(self.graph[i]) == 1)[0]

    def setVariables(self, robotswarm=None):
        if robotswarm is None or self.num_uav == 0:
            self.aux_var = SX.sym('aux_var')
            self.state_var = SX.sym('state_var')
            self.ctrl_var = SX.sym('ctrl_var')
        else:
            self.aux_var = [None] * self.num_uav
            self.n_aux_var = [None] * self.num_uav

            self.state_var = [None] * self.num_uav
            self.n_state_var = [None] * self.num_uav
            self.state_bd = [None] * self.num_uav

            self.ctrl_var = [None] * self.num_uav
            self.n_ctrl_var = [None] * self.num_uav
            self.ctrl_bd = [None] * self.num_uav

            self.neighbor_state_var = [None] * self.num_uav
            self.n_neighbor_state_var = [None] * self.num_uav

            for i in range(self.num_uav):
                self.aux_var[i] = vertcat(robotswarm[i].dyn_auxvar, robotswarm[i].cost_auxvar)
                self.n_aux_var[i] = self.aux_var[i].numel()
                self.state_var[i] = robotswarm[i].states
                self.n_state_var[i] = self.state_var[i].numel()
                self.state_bd[i] = robotswarm[i].states_bd
                self.ctrl_var[i] = robotswarm[i].ctrl
                self.n_ctrl_var[i] = self.ctrl_var[i].numel()
                self.ctrl_bd[i] = robotswarm[i].ctrl_bd

            for i in range(self.num_uav):
                neighbor_vec = []
                for j in self.neighbors[i]:
                    neighbor_vec += [self.state_var[j]]
                self.neighbor_state_var[i] = vcat(neighbor_vec)
                self.n_neighbor_state_var[i] = self.neighbor_state_var[i].numel()

    def setDyn(self, robotswarm):
        if not hasattr(self, 'aux_var'):
            sys.exit('Variables not set')

        self.dyn = [None] * self.num_uav
        self.dyn_fn = [None] * self.num_uav

        for i in range(self.num_uav):
            self.dyn[i] = robotswarm[i].states + robotswarm[i].dt * robotswarm[i].dynf
            self.dyn_fn[i] = casadi.Function('dynamics' + str(i),
                                             [self.state_var[i], self.ctrl_var[i], self.aux_var[i]], [self.dyn[i]])

    def setCosts(self, robotswarm):
        if not hasattr(self, 'aux_var'):
            sys.exit('Variables not set')

        assert robotswarm[0].path_cost.numel() == 1, "path_cost must be a scalar function"

        self.path_cost = [None] * self.num_uav
        self.path_cost_fn = [None] * self.num_uav
        self.final_cost = [None] * self.num_uav
        self.final_cost_fn = [None] * self.num_uav
        for i in range(self.num_uav):
            self.path_cost[i] = robotswarm[i].path_cost
            self.path_cost_fn[i] = casadi.Function('path_cost' + str(i),
                                                   [self.neighbor_state_var[i], self.ctrl_var[i],
                                                    self.aux_var[i]],
                                                   [self.path_cost[i]])
            self.final_cost[i] = robotswarm[i].final_cost
            self.final_cost_fn[i] = casadi.Function('final_cost' + str(i),
                                                    [self.neighbor_state_var[i], self.aux_var[i]],
                                                    [self.final_cost[i]])

    def def_DPMP(self):
        assert hasattr(self, 'state_var'), "Define the state variable first!"
        assert hasattr(self, 'ctrl_var'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(self, 'path_cost'), "Define the running cost function first!"
        assert hasattr(self, 'final_cost'), "Define the final cost function first!"

        # Define Theta as the parameter of the Game
        full_para = []
        for i in range(self.num_uav):
            full_para += [self.aux_var[i]]
        self.full_aux = vcat(full_para)

        # Define the Hamiltonian function
        self.costate = [None] * self.num_uav
        self.path_Hamil = [None] * self.num_uav
        self.final_Hamil = [None] * self.num_uav

        for i in range(self.num_uav):
            self.costate[i] = casadi.SX.sym('lambda' + str(i), self.state_var[i].numel())
            self.path_Hamil[i] = self.path_cost[i] + dot(self.dyn[i], self.costate[i])  # path Hamiltonian
            self.final_Hamil[i] = self.final_cost[i]  # final Hamiltonian

        # PDP equations
        # First-order derivative of path Hamiltonian
        self.dHx = [None] * self.num_uav
        self.dHx_fn = [None] * self.num_uav
        self.dHu = [None] * self.num_uav
        self.dHu_fn = [None] * self.num_uav

        for i in range(self.num_uav):
            self.dHx[i] = jacobian(self.path_Hamil[i], self.state_var[i]).T
            self.dHx_fn[i] = casadi.Function('dHx' + str(i),
                                             [self.neighbor_state_var[i], self.ctrl_var[i], self.costate[i],
                                              self.aux_var[i]],
                                             [self.dHx[i]])
            self.dHu[i] = jacobian(self.path_Hamil[i], self.ctrl_var[i]).T
            self.dHu_fn[i] = casadi.Function('dHu' + str(i),
                                             [self.neighbor_state_var[i], self.ctrl_var[i], self.costate[i],
                                              self.aux_var[i]],
                                             [self.dHu[i]])

        # First-order derivative of final Hamiltonian
        self.dhx = [None] * self.num_uav
        self.dhx_fn = [None] * self.num_uav

        for i in range(self.num_uav):
            self.dhx[i] = jacobian(self.final_Hamil[i], self.state_var[i]).T
            self.dhx_fn[i] = casadi.Function('dhx' + str(i),
                                             [self.neighbor_state_var[i], self.aux_var[i]],
                                             [self.dhx[i]])

        # Differentiating dynamics; notations here are consistent with the PDP paper
        self.M_l = [None] * self.num_uav
        self.M_l_fn = [None] * self.num_uav
        self.N_l = [None] * self.num_uav
        self.N_l_fn = [None] * self.num_uav
        self.C_l = [None] * self.num_uav
        self.C_l_fn = [None] * self.num_uav

        for i in range(self.num_uav):
            self.M_l[i] = jacobian(self.dyn[i], self.state_var[i])
            self.M_l_fn[i] = casadi.Function('M_l' + str(i), [self.state_var[i], self.ctrl_var[i], self.aux_var[i]],
                                             [self.M_l[i]])
            self.N_l[i] = jacobian(self.dyn[i], self.ctrl_var[i])
            self.N_l_fn[i] = casadi.Function('N_l' + str(i), [self.state_var[i], self.ctrl_var[i], self.aux_var[i]],
                                             [self.N_l[i]])
            self.C_l[i] = jacobian(self.dyn[i], self.full_aux)
            self.C_l_fn[i] = casadi.Function('C_l' + str(i), [self.state_var[i], self.ctrl_var[i], self.aux_var[i]],
                                             [self.C_l[i]])

        self.M_u = [None] * self.num_uav
        self.M_u_fn = [None] * self.num_uav
        self.N_u = [None] * self.num_uav
        self.N_u_fn = [None] * self.num_uav
        self.Q_u = [None] * self.num_uav
        self.Q_u_fn = [None] * self.num_uav
        self.S_u = [None] * self.num_uav
        self.S_u_fn = [None] * self.num_uav
        self.C_u = [None] * self.num_uav
        self.C_u_fn = [None] * self.num_uav

        for i in range(self.num_uav):
            self.M_u[i] = jacobian(self.dHu[i], self.state_var[i])
            self.M_u_fn[i] = casadi.Function('M_u' + str(i),
                                             [self.neighbor_state_var[i], self.ctrl_var[i], self.costate[i],
                                              self.aux_var[i]],
                                             [self.M_u[i]])
            self.N_u[i] = jacobian(self.dHu[i], self.ctrl_var[i])
            self.N_u_fn[i] = casadi.Function('N_u' + str(i),
                                             [self.neighbor_state_var[i], self.ctrl_var[i], self.costate[i],
                                              self.aux_var[i]],
                                             [self.N_u[i]])

            self.Q_u[i] = [None] * self.num_uav
            self.Q_u_fn[i] = [None] * self.num_uav
            for j in self.neighbors[i]:
                self.Q_u[i][j] = jacobian(self.dHu[i], self.state_var[j])
                self.Q_u_fn[i][j] = casadi.Function('Q_u' + str(i) + str(j),
                                                    [self.neighbor_state_var[i], self.ctrl_var[i], self.costate[i],
                                                     self.aux_var[i]],
                                                    [self.Q_u[i][j]])
            # Q_u[i][i] = M_u[i]

            self.S_u[i] = jacobian(self.dHu[i], self.costate[i])
            self.S_u_fn[i] = casadi.Function('S_u' + str(i),
                                             [self.neighbor_state_var[i], self.ctrl_var[i], self.costate[i],
                                              self.aux_var[i]],
                                             [self.S_u[i]])
            self.C_u[i] = jacobian(self.dHu[i], self.full_aux)
            self.C_u_fn[i] = casadi.Function('C_u' + str(i),
                                             [self.neighbor_state_var[i], self.ctrl_var[i], self.costate[i],
                                              self.aux_var[i]],
                                             [self.C_u[i]])

        self.M_x = [None] * self.num_uav
        self.M_x_fn = [None] * self.num_uav
        self.N_x = [None] * self.num_uav
        self.N_x_fn = [None] * self.num_uav
        self.Q_x = [None] * self.num_uav
        self.Q_x_fn = [None] * self.num_uav
        self.S_x = [None] * self.num_uav
        self.S_x_fn = [None] * self.num_uav
        self.C_x = [None] * self.num_uav
        self.C_x_fn = [None] * self.num_uav

        for i in range(self.num_uav):
            self.M_x[i] = jacobian(self.dHx[i], self.state_var[i])
            self.M_x_fn[i] = casadi.Function('M_x' + str(i),
                                             [self.neighbor_state_var[i], self.ctrl_var[i], self.costate[i],
                                              self.aux_var[i]],
                                             [self.M_x[i]])
            self.N_x[i] = jacobian(self.dHx[i], self.ctrl_var[i])
            self.N_x_fn[i] = casadi.Function('N_x' + str(i),
                                             [self.neighbor_state_var[i], self.ctrl_var[i], self.costate[i],
                                              self.aux_var[i]],
                                             [self.N_x[i]])

            self.Q_x[i] = [None] * self.num_uav
            self.Q_x_fn[i] = [None] * self.num_uav
            for j in self.neighbors[i]:
                self.Q_x[i][j] = jacobian(self.dHx[i], self.state_var[j])
                self.Q_x_fn[i][j] = casadi.Function('Q_x' + str(i) + str(j),
                                                    [self.neighbor_state_var[i], self.ctrl_var[i], self.costate[i],
                                                     self.aux_var[i]],
                                                    [self.Q_x[i][j]])
            # Q_x[i][i] = M_x[i]

            self.S_x[i] = jacobian(self.dHx[i], self.costate[i])
            self.S_x_fn[i] = casadi.Function('S_x' + str(i),
                                             [self.neighbor_state_var[i], self.ctrl_var[i], self.costate[i],
                                              self.aux_var[i]],
                                             [self.S_x[i]])
            self.C_x[i] = jacobian(self.dHx[i], self.full_aux)
            self.C_x_fn[i] = casadi.Function('C_x' + str(i),
                                             [self.neighbor_state_var[i], self.ctrl_var[i], self.costate[i],
                                              self.aux_var[i]],
                                             [self.C_x[i]])

        self.MT_x = [None] * self.num_uav
        self.MT_x_fn = [None] * self.num_uav
        self.CT_x = [None] * self.num_uav
        self.CT_x_fn = [None] * self.num_uav

        for i in range(self.num_uav):
            self.MT_x[i] = jacobian(self.dhx[i], self.state_var[i])
            self.MT_x_fn[i] = casadi.Function('MT_x' + str(i), [self.neighbor_state_var[i], self.aux_var[i]],
                                              [self.MT_x[i]])

            self.CT_x[i] = jacobian(self.dhx[i], self.full_aux)
            self.CT_x_fn[i] = casadi.Function('CT_x' + str(i), [self.neighbor_state_var[i], self.aux_var[i]],
                                              [self.CT_x[i]])

    def dyn_game_Solver(self, ini_state, horizon, auxvar_value, dt, loaded_traj=[None], gamma=1e-4, print_level=0,
                        eph=1, init_option=0, print_gap = 1e2, solver_option='d'):
        assert hasattr(self, 'state_var'), "Define the state variable first!"
        assert hasattr(self, 'ctrl_var'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(self, 'path_cost'), "Define the running cost function first!"
        assert hasattr(self, 'final_cost'), "Define the final cost function first!"

        # if type(ini_state) == numpy.ndarray:
        #    ini_state = ini_state.flatten().tolist()
        # print(auxvar_value)
        #
        if init_option == 1:
            time = numpy.array([k for k in range(horizon)])
            ini_tau_ctrl_traj = [None] * self.num_uav
            tau_ctrl_traj = [None] * self.num_uav
            tau_state_traj = [None] * self.num_uav
            tau_neighbor_state_traj = [None] * self.num_uav
            tau_costate_traj = [None] * self.num_uav
            d_tau_ctrl_traj = [None] * self.num_uav
            cost_full = [0] * self.num_uav

            for i in range(self.num_uav):
                ini_tau_ctrl_traj[i] = np.random.rand(horizon - 1, self.n_ctrl_var[i])

                d_tau_ctrl_traj[i] = np.zeros((horizon - 1, self.n_ctrl_var[i]))

                # ini_tau_ctrl_traj[i] = sample_init_ctrl
                tau_state_traj[i] = np.zeros((horizon, self.n_state_var[i]))
                tau_costate_traj[i] = np.zeros((horizon, self.n_state_var[i]))
                tau_neighbor_state_traj[i] = np.zeros((horizon, self.n_neighbor_state_var[i]))
                tau_ctrl_traj[i] = ini_tau_ctrl_traj[i]
                tau_state_traj[i][0, :] = ini_state[i]

            for tau in range(int(min(eph,1e3))):

                for i in range(self.num_uav):
                    # compute state trajectory
                    for k in range(horizon - 1):
                        tau_state_traj[i][k + 1, :] = self.dyn_fn[i](tau_state_traj[i][k, :], tau_ctrl_traj[i][k, :],
                                                                     auxvar_value[i]).T

                # Acquire neighbor trajectories
                for i in range(self.num_uav):
                    tau_neighbor_state_traj[i] = []
                    for j in self.neighbors[i]:
                        if len(tau_neighbor_state_traj[i]) == 0:
                            tau_neighbor_state_traj[i] = tau_state_traj[j]
                        else:
                            tau_neighbor_state_traj[i] = np.hstack((tau_neighbor_state_traj[i], tau_state_traj[j]))

                for i in range(self.num_uav):
                    tau_costate_traj[i][-1, :] = self.dhx_fn[i](tau_neighbor_state_traj[i][-1, :],
                                                                auxvar_value[i]).T
                    for k in range(horizon - 2, -1, -1):
                        tau_costate_traj[i][k, :] = self.dHx_fn[i](tau_neighbor_state_traj[i][k, :],
                                                                   tau_ctrl_traj[i][k, :],
                                                                   tau_costate_traj[i][k + 1, :], auxvar_value[i]).T

                    for k in range(horizon - 1):
                        d_tau_ctrl_traj[i][k, :] = self.dHu_fn[i](tau_neighbor_state_traj[i][k, :],
                                                                  tau_ctrl_traj[i][k, :],
                                                                  tau_costate_traj[i][k, :], auxvar_value[i]).T



                for i in range(self.num_uav):
                    tau_ctrl_traj[i] = tau_ctrl_traj[i] -  gamma * d_tau_ctrl_traj[i]


                for i in range(self.num_uav):
                    cost_full[i] = 0
                    for k in range(horizon - 1):
                        cost_full[i] = cost_full[i] + self.path_cost_fn[i](tau_neighbor_state_traj[i][k, :],
                                                                           tau_ctrl_traj[i][k, :], auxvar_value[i])
                    cost_full[i] = cost_full[i] + self.final_cost_fn[i](tau_neighbor_state_traj[i][k, :],
                                                                        auxvar_value[i])
                if tau % (print_gap) == 0:
                    # print(self.dHu_fn[0])
                    # print(self.dHu[0])
                    # print(tau_state_traj[2])
                    # print('here')
                    # print(tau_ctrl_traj[0])
                    # print(tau_costate_traj[2])
                    # print(d_tau_ctrl_traj[0])
                    print(tau, cost_full)
            #print(cost_full)

        if solver_option == 'd':
            adaptive_gain = 1
            hcost = [-10] * self.num_uav
            if init_option == 0:
                time = numpy.array([k for k in range(horizon)])
                ini_tau_ctrl_traj = [None] * self.num_uav
                tau_ctrl_traj = [None] * self.num_uav
                tau_state_traj = [None] * self.num_uav
                tau_neighbor_state_traj = [None] * self.num_uav
                tau_costate_traj = [None] * self.num_uav
                d_tau_ctrl_traj = [None] * self.num_uav
                cost_full = [0] * self.num_uav


            for i in range(self.num_uav):
                ini_tau_ctrl_traj[i] = np.random.rand(horizon - 1, self.n_ctrl_var[i])
                if init_option == 0:
                    if loaded_traj['control_traj_opt'][i].shape == ini_tau_ctrl_traj[i].shape:
                        ini_tau_ctrl_traj[i] = loaded_traj['control_traj_opt'][i]
                        # print('his_traj' + str(i) + ' loaded')
                    else:
                        print('Loaded traj Error')
                else:
                    ini_tau_ctrl_traj[i] = tau_ctrl_traj[i]

                d_tau_ctrl_traj[i] = np.zeros((horizon - 1, self.n_ctrl_var[i]))

                # ini_tau_ctrl_traj[i] = sample_init_ctrl
                tau_state_traj[i] = np.zeros((horizon, self.n_state_var[i]))
                tau_costate_traj[i] = np.zeros((horizon, self.n_state_var[i]))
                tau_neighbor_state_traj[i] = np.zeros((horizon, self.n_neighbor_state_var[i]))
                tau_ctrl_traj[i] = ini_tau_ctrl_traj[i]
                tau_state_traj[i][0, :] = ini_state[i]

            for tau in range(int(eph)):

                for i in range(self.num_uav):
                    # compute state trajectory
                    for k in range(horizon - 1):
                        tau_state_traj[i][k + 1, :] = self.dyn_fn[i](tau_state_traj[i][k, :], tau_ctrl_traj[i][k, :],
                                                                     auxvar_value[i]).T

                # Acquire neighbor trajectories
                for i in range(self.num_uav):
                    tau_neighbor_state_traj[i] = []
                    for j in self.neighbors[i]:
                        if len(tau_neighbor_state_traj[i]) == 0:
                            tau_neighbor_state_traj[i] = tau_state_traj[j]
                        else:
                            tau_neighbor_state_traj[i] = np.hstack((tau_neighbor_state_traj[i], tau_state_traj[j]))

                for i in range(self.num_uav):
                    tau_costate_traj[i][-1, :] = self.dhx_fn[i](tau_neighbor_state_traj[i][-1, :],
                                                                auxvar_value[i]).T
                    for k in range(horizon - 2, -1, -1):
                        tau_costate_traj[i][k, :] = self.dHx_fn[i](tau_neighbor_state_traj[i][k, :],
                                                                   tau_ctrl_traj[i][k, :],
                                                                   tau_costate_traj[i][k + 1, :], auxvar_value[i]).T

                    for k in range(horizon - 1):
                        d_tau_ctrl_traj[i][k, :] = self.dHu_fn[i](tau_neighbor_state_traj[i][k, :],
                                                                  tau_ctrl_traj[i][k, :],
                                                                  tau_costate_traj[i][k+1, :], auxvar_value[i]).T

                #if np.isnan(np.vstack(d_tau_ctrl_traj)).any():
                #    print('Error, ended with NAN')
                #    break
                #else:
                for i in range(self.num_uav):
                    tau_ctrl_traj[i] = tau_ctrl_traj[i] - adaptive_gain * gamma * d_tau_ctrl_traj[i]


                for i in range(self.num_uav):
                    cost_full[i] = 0
                    for k in range(horizon - 1):
                        cost_full[i] = cost_full[i] + self.path_cost_fn[i](tau_neighbor_state_traj[i][k, :],
                                                                           tau_ctrl_traj[i][k, :], auxvar_value[i])
                    cost_full[i] = cost_full[i] + self.final_cost_fn[i](tau_neighbor_state_traj[i][k, :],
                                                            auxvar_value[i])

                dif_cost = [cost_full[i] - hcost[i] for i in range(self.num_uav)]

                if np.linalg.norm(dif_cost) < 1e-12:
                    break
                hcost = cost_full[:]

                if (tau+1) % (print_gap) == 0:
                #    adaptive_gain = (1e3)/ (1e3+tau)
                    # print(self.dHu_fn[0])
                    # print(self.dHu[0])
                    # print(tau_state_traj[2])
                    # print('here')
                    # print(tau_ctrl_traj[0])
                    # print(tau_costate_traj[2])
                    # print(d_tau_ctrl_traj[0])
                    print(tau, adaptive_gain, cost_full)
                    #print(dif_cost)
                #    sio.savemat('data/u_gap.mat',
                #                {'d_tau_ctrl_traj': d_tau_ctrl_traj,'tau_state_traj':tau_state_traj,'tau_ctrl_traj':tau_ctrl_traj,'tau_costate_traj':tau_costate_traj})

                #print(d_tau_ctrl_traj)
                #exit()
                opt_sol = {"state_traj_opt": tau_state_traj,
                           "control_traj_opt": tau_ctrl_traj,
                           "costate_traj_opt": tau_costate_traj,
                           'auxvar_value': auxvar_value,
                           "time": time,
                           "horizon": horizon,
                           "cost": cost_full}
                # if tau % (1e3) == 0:
                #    sio.savemat('data/uav_formation_demos.mat', {'trajectories': opt_sol,
                #                           'dt': dt,
                #                           'true_parameter': auxvar_value})
            print(cost_full)

        elif solver_option == 'c':

            time = numpy.array([k for k in range(horizon)])
            ini_sol_ctrl_traj = [None] * self.num_uav
            ini_sol_state_traj = [None] * self.num_uav
            ini_sol_costate_traj = [None] * self.num_uav

            for i in range(self.num_uav):
                if init_option == 0:
                    ini_sol_ctrl_traj[i] = loaded_traj['control_traj_opt'][i]
                    ini_sol_state_traj[i] = loaded_traj['state_traj_opt'][i]
                    ini_sol_costate_traj[i] = loaded_traj['costate_traj_opt'][i]
                    # print('his_traj' + str(i) + ' loaded')

                else:
                    ini_sol_ctrl_traj[i] = tau_ctrl_traj[i]
                    ini_sol_state_traj[i] = tau_state_traj[i]
                    ini_sol_costate_traj[i] = tau_costate_traj[i]

            w_stac = []
            g_stac = []
            var_ctrl_traj = [None] * self.num_uav
            var_state_traj = [None] * self.num_uav
            var_costate_traj = [None] * self.num_uav
            for i in range(self.num_uav):
                var_ctrl_traj[i] = [None] * (horizon - 1)
                var_state_traj[i] = [None] * horizon
                var_costate_traj[i] = [None] * horizon
                for k in range(horizon):
                    if k < (horizon - 1):
                        var_ctrl_traj[i][k] = MX.sym('U_' + str(i) + str(k), self.n_ctrl_var[i])
                    var_state_traj[i][k] = MX.sym('X_' + str(i) + str(k), self.n_state_var[i])
                    var_costate_traj[i][k] = MX.sym('L_' + str(i) + str(k), self.n_state_var[i])
                for k in range(horizon):
                    w_stac += [var_state_traj[i][k]]
                for k in range(horizon-1):
                    w_stac += [var_ctrl_traj[i][k]]
                for k in range(horizon):
                    w_stac += [var_costate_traj[i][k]]

            w = vertcat(*w_stac)

            var_neighbor_state_traj = [None] * self.num_uav
            for i in range(self.num_uav):
                var_neighbor_state_traj[i] = [None] * horizon
                for k in range(horizon):
                    var_neighbor_vec = []
                    for j in self.neighbors[i]:
                        var_neighbor_vec += [var_state_traj[j][k]]
                    var_neighbor_state_traj[i][k] = vcat(var_neighbor_vec)

            for i in range(self.num_uav):
                g_stac += [var_state_traj[i][0] - ini_state[i]]
                for k in range(horizon - 1):
                    g_stac += [var_state_traj[i][k + 1] - self.dyn_fn[i](var_state_traj[i][k], var_ctrl_traj[i][k],
                                                                         auxvar_value[i])]
                    g_stac += [
                        self.dHu_fn[i](var_neighbor_state_traj[i][k], var_ctrl_traj[i][k], var_costate_traj[i][k + 1],
                                       auxvar_value[i])]
                    g_stac += [var_costate_traj[i][k] -
                               self.dHx_fn[i](var_neighbor_state_traj[i][k], var_ctrl_traj[i][k],
                                              var_costate_traj[i][k + 1], auxvar_value[i])]
                g_stac += [var_costate_traj[i][-1] - self.dhx_fn[i](var_neighbor_state_traj[i][-1], auxvar_value[i])]
            g = vertcat(*g_stac)
            #print(g)
            #print(w)

            #opts = {'error_on_fail': True, 'print_iteration': True}
            opts = {'nlpsol':'ipopt'}
            prob = {'x': w, 'g': g}
            #solver = rootfinder('G', 'newton', prob, opts)
            solver = rootfinder('G', 'nlpsol', prob, opts)
            print('Start solving the trajectory:')
            ini_sol_stac = []
            for i in range(self.num_uav):
                ini_sol_stac += [np.hstack(ini_sol_state_traj[i])]
                ini_sol_stac += [np.hstack(ini_sol_ctrl_traj[i])]
                ini_sol_stac += [np.hstack(ini_sol_costate_traj[i])]
            ini_sol = np.hstack(ini_sol_stac)

            sol = solver(ini_sol, [])

            full_traj = np.reshape(sol, (self.num_uav,-1))

            #print(full_traj.shape)
            decomposed_traj = np.split(full_traj, [self.n_state_var[0] * horizon,
                                               self.n_state_var[0] * horizon + self.n_ctrl_var[0] * (horizon - 1)], 1)
            #print(decomposed_traj[1].shape)

            sol_state_traj = [np.reshape(decomposed_traj[0][i],(horizon,-1) ) for i in range(self.num_uav)]
            sol_ctrl_traj = [np.reshape(decomposed_traj[1][i], (horizon-1, -1)) for i in range(self.num_uav)]
            sol_costate_traj = [np.reshape(decomposed_traj[2][i], (horizon, -1)) for i in range(self.num_uav)]
            opt_sol = {"state_traj_opt": sol_state_traj,
                       "control_traj_opt": sol_ctrl_traj,
                       "costate_traj_opt": sol_costate_traj,
                       'auxvar_value': auxvar_value,
                       "time": time,
                       "horizon": horizon,
                       "cost": 0}

        elif solver_option == 'cs':

            time = numpy.array([k for k in range(horizon)])
            ini_sol_ctrl_traj = [None] * self.num_uav
            ini_sol_state_traj = [None] * self.num_uav
            ini_sol_costate_traj = [None] * self.num_uav

            for i in range(self.num_uav):
                if init_option == 0:
                    ini_sol_ctrl_traj[i] = loaded_traj['control_traj_opt'][i]
                    ini_sol_state_traj[i] = loaded_traj['state_traj_opt'][i]

                    ini_sol_costate_traj[i] = np.zeros((horizon, self.n_state_var[i]))
                    # print('his_traj' + str(i) + ' loaded')

                else:
                    ini_sol_ctrl_traj[i] = tau_ctrl_traj[i]
                    ini_sol_state_traj[i] = tau_state_traj[i]
                    ini_sol_costate_traj[i] = tau_costate_traj[i]

            w_stac = []
            g_stac = []
            var_ctrl_traj = [None] * self.num_uav
            var_state_traj = [None] * self.num_uav
            var_costate_traj = [None] * self.num_uav
            for i in range(self.num_uav):
                var_ctrl_traj[i] = [None] * (horizon - 1)
                var_state_traj[i] = [None] * horizon
                var_costate_traj[i] = [None] * horizon
                for k in range(horizon-1):
                    var_ctrl_traj[i][k] = MX.sym('U_' + str(i) + str(k), self.n_ctrl_var[i])
                    #var_state_traj[i][k] = MX.sym('X_' + str(i) + str(k), self.n_state_var[i])
                    #var_costate_traj[i][k] = MX.sym('L_' + str(i) + str(k), self.n_state_var[i])
                #for k in range(horizon):
                    #w_stac += [var_state_traj[i][k]]
                for k in range(horizon-1):
                    w_stac += [var_ctrl_traj[i][k]]
                #for k in range(horizon):
                    #w_stac += [var_costate_traj[i][k]]

            w = vertcat(*w_stac)


            for i in range(self.num_uav):
                var_state_traj[i][0] = ini_state[i]
                for k in range(horizon - 1):
                    var_state_traj[i][k + 1] = self.dyn_fn[i](var_state_traj[i][k], var_ctrl_traj[i][k],
                                                                 auxvar_value[i])

            var_neighbor_state_traj = [None] * self.num_uav
            for i in range(self.num_uav):
                var_neighbor_state_traj[i] = [None] * horizon
                for k in range(horizon):
                    var_neighbor_vec = []
                    for j in self.neighbors[i]:
                        var_neighbor_vec += [var_state_traj[j][k]]
                    var_neighbor_state_traj[i][k] = vertcat(*var_neighbor_vec)

            for i in range(self.num_uav):
                var_costate_traj[i][-1] = self.dhx_fn[i](var_neighbor_state_traj[i][-1],
                                                            auxvar_value[i])
                for k in range(horizon - 2, -1, -1):
                    var_costate_traj[i][k] = self.dHx_fn[i](var_neighbor_state_traj[i][k], var_ctrl_traj[i][k],
                                   var_costate_traj[i][k + 1], auxvar_value[i])

                for k in range(horizon - 1):
                    g_stac += [self.dHu_fn[i](var_neighbor_state_traj[i][k], var_ctrl_traj[i][k], var_costate_traj[i][k + 1],
                               auxvar_value[i])]


            g = vertcat(*g_stac)
            #print('shape')
            #print(g.shape)
            #print(w)

            #opts = {'error_on_fail': True, 'print_iteration': True}
            opts = {'nlpsol':'ipopt'}
            #opts = {'nlpsol':'ipopt', 'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
            prob = {'x': w, 'g': g}
            #solver = rootfinder('G', 'newton', prob, opts)
            solver = rootfinder('G', 'nlpsol', prob, opts)
            # print('Start solving the trajectory:')
            ini_sol_stac = []
            for i in range(self.num_uav):
                #ini_sol_stac += [np.hstack(ini_sol_state_traj[i])]
                ini_sol_stac += [np.hstack(ini_sol_ctrl_traj[i])]
                #ini_sol_stac += [np.hstack(ini_sol_costate_traj[i])]
            ini_sol = np.hstack(ini_sol_stac)

            if print_level==0:
                sys.stdout = open(os.devnull, "w")

            sol = solver(ini_sol, [])

            sys.stdout = sys.__stdout__

            full_traj = np.reshape(sol, (self.num_uav,-1))

            # Check solution
            # g_fuc=casadi.Function('g_equation', [w], [g])
            # print(g_fuc(sol))


            #sol_state_traj = [np.reshape(decomposed_traj[0][i],(horizon,-1) ) for i in range(self.num_uav)]
            sol_ctrl_traj = [np.reshape(full_traj[i], (horizon-1, -1)) for i in range(self.num_uav)]
            #sol_costate_traj = [np.reshape(decomposed_traj[2][i], (horizon, -1)) for i in range(self.num_uav)]

            opt_ctrl_traj = [None]* self.num_uav
            opt_state_traj = [None]* self.num_uav
            opt_costate_traj = [None]* self.num_uav
            opt_neighbor_state_traj = [None]* self.num_uav
            for i in range(self.num_uav):
                opt_state_traj[i] = np.zeros((horizon, self.n_state_var[i]))
                opt_ctrl_traj[i] = np.zeros((horizon-1, self.n_ctrl_var[i]))
                opt_costate_traj[i] = np.zeros((horizon, self.n_state_var[i]))
                opt_state_traj[i][0,:] = ini_state[i]
                for k in range(horizon - 1):
                    opt_ctrl_traj[i][k,:] = sol_ctrl_traj[i][k,:]
                    opt_state_traj[i][k + 1, :] = self.dyn_fn[i](opt_state_traj[i][k], opt_ctrl_traj[i][k],
                                                                 auxvar_value[i]).T

            for i in range(self.num_uav):
                opt_neighbor_state_traj[i] = [None] * horizon
                for k in range(horizon):
                    opt_neighbor_vec = []
                    for j in self.neighbors[i]:
                        opt_neighbor_vec += [opt_state_traj[j][k]]
                    opt_neighbor_state_traj[i][k] = vertcat(*opt_neighbor_vec)

            for i in range(self.num_uav):
                opt_costate_traj[i][-1, :] = self.dhx_fn[i](opt_neighbor_state_traj[i][-1],
                                                         auxvar_value[i]).T
                for k in range(horizon - 2, -1, -1):
                    opt_costate_traj[i][k,:] = self.dHx_fn[i](opt_neighbor_state_traj[i][k], opt_ctrl_traj[i][k],
                                   opt_costate_traj[i][k + 1], auxvar_value[i]).T

            opt_sol = {"state_traj_opt": opt_state_traj,
                       "control_traj_opt": opt_ctrl_traj,
                       "costate_traj_opt": opt_costate_traj,
                       'auxvar_value': auxvar_value,
                       "time": time,
                       "horizon": horizon,
                       "cost": 'Solved'}

        return opt_sol



    def D_PMP_Solver(self, horizon, auxvar_value, gamma=1e-4, loaded_traj=[None], print_level=0, eph=1,
                     solover_option=0):

        curr_ctrl_traj = [None] * self.num_uav
        curr_state_traj = [None] * self.num_uav
        curr_costate_traj = [None] * self.num_uav
        curr_neighbor_state_traj = [None] * self.num_uav


        for i in range(self.num_uav):
            curr_ctrl_traj[i] = np.random.rand(horizon - 1, self.n_ctrl_var[i])
            if loaded_traj['control_traj_opt'][i].shape == curr_ctrl_traj[i].shape:
                curr_state_traj[i] = loaded_traj['state_traj_opt'][i]
                curr_ctrl_traj[i] = loaded_traj['control_traj_opt'][i]
                curr_costate_traj[i] = loaded_traj['costate_traj_opt'][i]
            else:
                print('Loaded traj Error')

        # ---
        #for i in range(self.num_uav):
        #    curr_neighbor_state_traj[i] = []
        #    for j in self.neighbors[i]:
        #        if len(curr_neighbor_state_traj[i]) == 0:
        #            curr_neighbor_state_traj[i] = curr_state_traj[j]
        #        else:
        #            curr_neighbor_state_traj[i] = np.hstack((curr_neighbor_state_traj[i], curr_state_traj[j]))

        for i in range(self.num_uav):
            curr_neighbor_state_traj_stac = []
            for j in self.neighbors[i]:
                curr_neighbor_state_traj_stac += [curr_state_traj[j]]
            curr_neighbor_state_traj[i] = np.hstack(curr_neighbor_state_traj_stac)

        # Direct linear equation solver
        if solover_option == 0:
            # diff dyn
            M_l_V = [None] * (horizon - 1)
            N_l_V = [None] * (horizon - 1)
            C_l_V = [None] * (horizon - 1)
            # diff input
            M_u_V = [None] * (horizon - 1)
            N_u_V = [None] * (horizon - 1)
            Q_u_V = [None] * (horizon - 1)
            S_u_V = [None] * (horizon - 1)
            C_u_V = [None] * (horizon - 1)
            # diff costate
            M_x_V = [None] * (horizon - 1)
            N_x_V = [None] * (horizon - 1)
            Q_x_V = [None] * (horizon - 1)
            S_x_V = [None] * (horizon - 1)
            C_x_V = [None] * (horizon - 1)
            # diff boundary
            MT_x_V = [None] * (horizon - 1)
            CT_x_V = [None] * (horizon - 1)
            for i in range(self.num_uav):
                MT_x_V[i] = self.MT_x_fn[i](curr_neighbor_state_traj[i][-1, :], auxvar_value[i])
                CT_x_V[i] = self.CT_x_fn[i](curr_neighbor_state_traj[i][-1, :], auxvar_value[i])

            for k in range(horizon - 1):
                # diff dyn
                M_l_V[k] = [None] * self.num_uav
                N_l_V[k] = [None] * self.num_uav
                C_l_V[k] = [None] * self.num_uav
                # diff input
                M_u_V[k] = [None] * self.num_uav
                N_u_V[k] = [None] * self.num_uav
                Q_u_V[k] = [None] * self.num_uav
                S_u_V[k] = [None] * self.num_uav
                C_u_V[k] = [None] * self.num_uav
                # diff costate
                M_x_V[k] = [None] * self.num_uav
                N_x_V[k] = [None] * self.num_uav
                Q_x_V[k] = [None] * self.num_uav
                S_x_V[k] = [None] * self.num_uav
                C_x_V[k] = [None] * self.num_uav

                for i in range(self.num_uav):
                    # diff dyn
                    M_l_V[k][i] = self.M_l_fn[i](curr_state_traj[i][k, :],
                                                 curr_ctrl_traj[i][k, :],
                                                 auxvar_value[i])
                    N_l_V[k][i] = self.N_l_fn[i](curr_state_traj[i][k, :],
                                                 curr_ctrl_traj[i][k, :],
                                                 auxvar_value[i])
                    C_l_V[k][i] = self.C_l_fn[i](curr_state_traj[i][k, :],
                                                 curr_ctrl_traj[i][k, :],
                                                 auxvar_value[i])


                    # diff input
                    M_u_V[k][i] = self.M_u_fn[i](curr_neighbor_state_traj[i][k, :],
                                                 curr_ctrl_traj[i][k, :],
                                                 curr_costate_traj[i][k, :], auxvar_value[i])
                    N_u_V[k][i] = self.N_u_fn[i](curr_neighbor_state_traj[i][k, :],
                                                 curr_ctrl_traj[i][k, :],
                                                 curr_costate_traj[i][k, :], auxvar_value[i])
                    Q_u_V[k][i] = [None] * self.num_uav
                    for j in self.neighbors[i]:
                        Q_u_V[k][i][j] = self.Q_u_fn[i][j](curr_neighbor_state_traj[i][k, :],
                                                           curr_ctrl_traj[i][k, :],
                                                           curr_costate_traj[i][k, :], auxvar_value[i])
                    S_u_V[k][i] = self.S_u_fn[i](curr_neighbor_state_traj[i][k, :],
                                                 curr_ctrl_traj[i][k, :],
                                                 curr_costate_traj[i][k, :], auxvar_value[i])
                    C_u_V[k][i] = self.C_u_fn[i](curr_neighbor_state_traj[i][k, :],
                                                 curr_ctrl_traj[i][k, :],
                                                 curr_costate_traj[i][k, :], auxvar_value[i])

                    # diff costate
                    M_x_V[k][i] = self.M_x_fn[i](curr_neighbor_state_traj[i][k, :],
                                                 curr_ctrl_traj[i][k, :],
                                                 curr_costate_traj[i][k, :], auxvar_value[i])

                    N_x_V[k][i] = self.N_x_fn[i](curr_neighbor_state_traj[i][k, :],
                                                 curr_ctrl_traj[i][k, :],
                                                 curr_costate_traj[i][k, :], auxvar_value[i])
                    Q_x_V[k][i] = [None] * self.num_uav
                    for j in self.neighbors[i]:
                        Q_x_V[k][i][j] = self.Q_x_fn[i][j](curr_neighbor_state_traj[i][k, :],
                                                           curr_ctrl_traj[i][k, :],
                                                           curr_costate_traj[i][k, :], auxvar_value[i])
                    S_x_V[k][i] = self.S_x_fn[i](curr_neighbor_state_traj[i][k, :],
                                                 curr_ctrl_traj[i][k, :],
                                                 curr_costate_traj[i][k, :], auxvar_value[i])
                    C_x_V[k][i] = self.C_x_fn[i](curr_neighbor_state_traj[i][k, :],
                                                 curr_ctrl_traj[i][k, :],
                                                 curr_costate_traj[i][k, :], auxvar_value[i])


            # def bar matrices
            B_M_dyn = [None] * self.num_uav
            B_N_dyn = [None] * self.num_uav
            B_C_dyn = [None] * self.num_uav

            B_M_input = [None] * self.num_uav
            B_N_input = [None] * self.num_uav
            B_Q_input = [None] * self.num_uav
            B_S_input = [None] * self.num_uav
            B_C_input = [None] * self.num_uav

            B_M_costate = [None] * self.num_uav
            B_N_costate = [None] * self.num_uav
            B_Q_costate = [None] * self.num_uav
            B_S_costate = [None] * self.num_uav
            B_C_costate = [None] * self.num_uav

            B_M_T = [None] * self.num_uav
            B_S_T = [None] * self.num_uav
            B_C_T = [None] * self.num_uav

            B_M_I = [None] * self.num_uav

            B_M = [None] * self.num_uav
            B_N = [None] * self.num_uav
            B_S = [None] * self.num_uav
            B_Q = [None] * self.num_uav
            B_C = [None] * self.num_uav

            for i in range(self.num_uav):
                # diff dyn
                stack_M = []
                for k in range(horizon - 1):
                    stack_M += [M_l_V[k][i]]
                B_M_dyn_M = np.hstack((spy.linalg.block_diag(*stack_M),
                                       np.zeros((self.n_state_var[i] * (horizon - 1), self.n_state_var[i]))))
                B_M_dyn_I = np.hstack((np.zeros((self.n_state_var[i] * (horizon - 1), self.n_state_var[i])),
                                       np.identity(self.n_state_var[i] * (horizon - 1))))
                B_M_dyn[i] = B_M_dyn_M - B_M_dyn_I

                stack_N = []
                for k in range(horizon - 1):
                    stack_N += [N_l_V[k][i]]
                B_N_dyn[i] = spy.linalg.block_diag(*stack_N)

                stack_C = []
                for k in range(horizon - 1):
                    stack_C += [C_l_V[k][i]]
                B_C_dyn[i] = np.vstack(stack_C)

                # diff input
                stack_M = []
                for k in range(horizon - 1):
                    stack_M += [M_u_V[k][i]]
                B_M_input[i] = np.hstack((spy.linalg.block_diag(*stack_M),
                                          np.zeros((self.n_ctrl_var[i] * (horizon - 1), self.n_state_var[i]))))

                stack_N = []
                for k in range(horizon - 1):
                    stack_N += [N_u_V[k][i]]
                B_N_input[i] = spy.linalg.block_diag(*stack_N)

                B_Q_input[i] = [None] * self.num_uav
                for j in self.neighbors[i]:
                    stack_Q = []
                    for k in range(horizon - 1):
                        stack_Q += [Q_u_V[k][i][j]]
                    B_Q_input[i][j] = np.hstack((spy.linalg.block_diag(*stack_Q),
                                                 np.zeros((self.n_ctrl_var[i] * (horizon - 1), self.n_state_var[i]))))

                stack_S = []
                for k in range(horizon - 1):
                    stack_S += [S_u_V[k][i]]
                B_S_input[i] = np.hstack((np.zeros((self.n_ctrl_var[i] * (horizon - 1), self.n_state_var[i])),
                                          spy.linalg.block_diag(*stack_S)))

                stack_C = []
                for k in range(horizon - 1):
                    stack_C += [C_u_V[k][i]]
                B_C_input[i] = np.vstack(stack_C)

                # diff costate
                stack_M = []
                for k in range(horizon - 1):
                    stack_M += [M_x_V[k][i]]
                B_M_costate[i] = np.hstack((spy.linalg.block_diag(*stack_M),
                                            np.zeros((self.n_state_var[i] * (horizon - 1), self.n_state_var[i]))))

                stack_N = []
                for k in range(horizon - 1):
                    stack_N += [N_x_V[k][i]]
                B_N_costate[i] = spy.linalg.block_diag(*stack_N)

                B_Q_costate[i] = [None] * self.num_uav
                for j in self.neighbors[i]:
                    stack_Q = []
                    for k in range(horizon - 1):
                        stack_Q += [Q_x_V[k][i][j]]
                    B_Q_costate[i][j] = np.hstack((spy.linalg.block_diag(*stack_Q),
                                                   np.zeros(
                                                       (self.n_state_var[i] * (horizon - 1), self.n_state_var[i]))))

                stack_S = []
                for k in range(horizon - 1):
                    stack_S += [S_x_V[k][i]]

                B_S_costate_M = np.hstack((np.zeros((self.n_state_var[i] * (horizon - 1), self.n_state_var[i])),
                                           spy.linalg.block_diag(*stack_S)))
                B_S_costate_I = np.hstack((np.identity(self.n_state_var[i] * (horizon - 1)),
                                           np.zeros((self.n_state_var[i] * (horizon - 1), self.n_state_var[i]))))
                B_S_costate[i] = B_S_costate_M - B_S_costate_I

                stack_C = []
                for k in range(horizon - 1):
                    stack_C += [C_x_V[k][i]]
                B_C_costate[i] = np.vstack(stack_C)

                # diff terminal
                B_M_T[i] = np.hstack((np.zeros((self.n_state_var[i], self.n_state_var[i] * (horizon - 1))), MT_x_V[i]))
                B_S_T[i] = -np.hstack((np.zeros((self.n_state_var[i], self.n_state_var[i] * (horizon - 1))),
                                       np.identity(self.n_state_var[i])))
                B_C_T[i] = CT_x_V[i]

                # diff initial
                B_M_I[i] = np.hstack([np.identity(self.n_state_var[i]),
                                      np.zeros((self.n_state_var[i], self.n_state_var[i] * (horizon - 1)))])

                # diff full stack
                B_M[i] = np.vstack([B_M_dyn[i], B_M_input[i], B_M_costate[i], B_M_T[i], B_M_I[i]])
                B_N[i] = np.vstack([B_N_dyn[i], B_N_input[i], B_N_costate[i],
                                    np.zeros([self.n_state_var[i], self.n_ctrl_var[i] * (horizon - 1)]),
                                    np.zeros([self.n_state_var[i], self.n_ctrl_var[i] * (horizon - 1)])])
                B_S[i] = np.vstack(
                    [np.zeros_like(B_M_dyn[i]), B_S_input[i], B_S_costate[i], B_S_T[i], np.zeros_like(B_S_T[i])])
                B_C[i] = np.vstack([B_C_dyn[i], B_C_input[i], B_C_costate[i], B_C_T[i], np.zeros_like(B_C_T[i])])

                B_Q[i] = [np.zeros_like(B_M[i])] * self.num_uav
                for j in self.neighbors[i]:
                    B_Q[i][j] = np.vstack(
                        [np.zeros_like(B_M_dyn[i]), B_Q_input[i][j], B_Q_costate[i][j], np.zeros_like(B_M_T[i]),
                         np.zeros_like(B_M_I[i])])

                # print(B_M[i].shape)
                # print(B_N[i].shape)
                # print(B_S[i].shape)
                # print(B_C[i].shape)
                # print(B_Q[i][0].shape)

            A_matrix = [None] * self.num_uav
            A_compact_stack = []
            C_compact_stack = []
            for i in range(self.num_uav):
                A_matrix[i] = [np.zeros_like(np.hstack([B_M[i], B_N[i], B_S[i]]))] * self.num_uav
                A_row_stack = []
                for j in range(self.num_uav):
                    if j == i:
                        A_matrix[i][j] = np.hstack([B_M[i], B_N[i], B_S[i]])
                    elif j in self.neighbors[i]:
                        A_matrix[i][j] = np.hstack([B_Q[i][j], np.zeros_like(B_N[i]), np.zeros_like(B_S[i])])
                    A_row_stack += [A_matrix[i][j]]
                A_compact_stack += [np.hstack(A_row_stack)]
                C_compact_stack += [B_C[i]]
            A_compact = np.vstack(A_compact_stack)
            C_compact = np.vstack(C_compact_stack)
            #sio.savemat('data/matrix_check.mat',
            #            {'A_compact': A_compact, 'C_compact':C_compact})

            Y_compact = np.linalg.solve(A_compact, -C_compact)

            Y_robot = np.vsplit(Y_compact, self.num_uav)

            # X_robot=[None]*self.num_uav
            # for i in range(self.num_uav):
            #    X_robot[i] = np.split(Y_robot[i], [self.n_state_var[0] * horizon,
            #                             self.n_state_var[0] * horizon + self.n_ctrl_var[0] * (horizon - 1)])

            Y_split_robot = np.split(Y_robot, [self.n_state_var[0] * horizon,
                                               self.n_state_var[0] * horizon + self.n_ctrl_var[0] * (horizon - 1)], 1)

            X_robot = np.split(Y_split_robot[0], horizon, 1)
            U_robot = np.split(Y_split_robot[1], horizon - 1, 1)

        return [X_robot, U_robot]
