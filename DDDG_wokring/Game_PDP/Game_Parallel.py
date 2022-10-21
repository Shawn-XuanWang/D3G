import sys
from casadi import *
import numpy as np
import scipy as spy
import scipy.io as sio

class DGSys:
    def __init__(self, robot_index=None, project_name="my dynamic game system"):
        self.project_name = project_name
        self.graph = None
        self.num_uav = None
        self.full_aux = None
        self.robot_index = robot_index

    def setGraph(self, num_robot=None, adj_matrix=None):
        if num_robot is None:
            sys.exit('Number of robots needs to be defined!')

        if adj_matrix is None or np.asarray(adj_matrix).shape != (num_robot, num_robot):
            sys.exit('Adjacency matrix needs to be properly defined!')
        self.graph = adj_matrix

        self.num_uav = num_robot

        self.neighbors = [None]
        self.n_neighbors = [None]
        for i in range(self.num_uav):
            self.neighbors = np.where(np.array(self.graph[i]) == 1)[0]
        self.n_neighbors = self.neighbors.shape

    def setVariables(self, uav_swarm=None):

        if uav_swarm is None or self.num_uav == 0:
            self.aux_var = SX.sym('aux_var')
            self.state_var = SX.sym('state_var')
            self.ctrl_var = SX.sym('ctrl_var')
        else:
            self.aux_var = [None]
            self.n_aux_var = [None]
            self.dt = [None]

            self.state_var = [None]
            self.n_state_var = [None]
            self.state_bd = [None]

            self.ctrl_var = [None]
            self.n_ctrl_var = [None]
            self.dynf = [None]
            self.neighbor_state_var = [None]
            self.n_neighbor_state_var = [None]

            self.dt = uav_swarm.dt
            self.aux_var = vertcat(uav_swarm.dyn_auxvar[self.robot_index], uav_swarm.cost_auxvar[self.robot_index])
            self.n_aux_var = self.aux_var.numel()
            self.state_var = uav_swarm.states[self.robot_index]
            self.n_state_var = self.state_var.numel()
            self.state_bd = uav_swarm.states_bd[self.robot_index]
            self.ctrl_var = uav_swarm.ctrl[self.robot_index]
            self.n_ctrl_var = self.ctrl_var.numel()
            self.dynf = uav_swarm.dynf[self.robot_index]

            neighbor_vec = []
            for j in self.neighbors:
                neighbor_vec += [uav_swarm.states[j]]
            self.neighbor_state_var = vcat(neighbor_vec)
            self.n_neighbor_state_var = self.neighbor_state_var.numel()

    def setDyn(self):
        if not hasattr(self, 'aux_var'):
            sys.exit('Variables not set')

        self.dyn = [None]
        self.dyn_fn = [None]

        self.dyn = self.state_var + self.dt * self.dynf
        self.dyn_fn = casadi.Function('dynamics' + str(self.robot_index),
                                      [self.state_var, self.ctrl_var, self.aux_var], [self.dyn])

    def setCosts(self, robotswarm):
        if not hasattr(self, 'aux_var'):
            sys.exit('Variables not set')

        assert robotswarm.path_cost[self.robot_index].numel() == 1, "path_cost must be a scalar function"

        self.path_cost = [None]
        self.path_cost_fn = [None]
        self.final_cost = [None]
        self.final_cost_fn = [None]

        self.path_cost = robotswarm.path_cost[self.robot_index]
        self.path_cost_fn = casadi.Function('path_cost' + str(self.robot_index),
                                               [self.neighbor_state_var, self.ctrl_var,
                                                self.aux_var],
                                               [self.path_cost])
        self.final_cost = robotswarm.final_cost[self.robot_index]
        self.final_cost_fn = casadi.Function('final_cost' + str(self.robot_index),
                                                [self.neighbor_state_var, self.aux_var],
                                                [self.final_cost])

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
        self.costate = [None]
        self.path_Hamil = [None]
        self.final_Hamil = [None]

        self.costate = casadi.SX.sym('lambda' + str(self.robot_index), self.state_var.numel())
        self.path_Hamil = self.path_cost + dot(self.dyn, self.costate)  # path Hamiltonian
        self.final_Hamil = self.final_cost  # final Hamiltonian

        # PDP equations
        # First-order derivative of path Hamiltonian
        self.dHx = [None] * self.num_uav
        self.dHx_fn = [None] * self.num_uav
        self.dHu = [None] * self.num_uav
        self.dHu_fn = [None] * self.num_uav

        self.dHx = jacobian(self.path_Hamil, self.state_var).T
        self.dHx_fn = casadi.Function('dHx' + str(self.robot_index),
                                         [self.neighbor_state_var, self.ctrl_var, self.costate,
                                          self.aux_var],
                                         [self.dHx])
        self.dHu = jacobian(self.path_Hamil, self.ctrl_var).T
        self.dHu_fn = casadi.Function('dHu' + str(self.robot_index),
                                         [self.neighbor_state_var, self.ctrl_var, self.costate,
                                          self.aux_var],
                                         [self.dHu])

        # First-order derivative of final Hamiltonian
        self.dhx = [None]
        self.dhx_fn = [None]

        self.dhx = jacobian(self.final_Hamil, self.state_var).T
        self.dhx_fn = casadi.Function('dhx' + str(self.robot_index),
                                         [self.neighbor_state_var, self.aux_var],
                                         [self.dhx])

        # Differentiating dynamics; notations here are consistent with the PDP paper
        self.M_l = [None]
        self.M_l_fn = [None]
        self.N_l = [None]
        self.N_l_fn = [None]
        self.C_l = [None]
        self.C_l_fn = [None]

        self.M_l = jacobian(self.dyn, self.state_var)
        self.M_l_fn = casadi.Function('M_l' + str(self.robot_index), [self.state_var, self.ctrl_var, self.aux_var],
                                         [self.M_l])
        self.N_l = jacobian(self.dyn, self.ctrl_var)
        self.N_l_fn = casadi.Function('N_l' + str(self.robot_index), [self.state_var, self.ctrl_var, self.aux_var],
                                         [self.N_l])
        self.C_l = jacobian(self.dyn, self.full_aux)
        self.C_l_fn = casadi.Function('C_l' + str(self.robot_index), [self.state_var, self.ctrl_var, self.aux_var],
                                         [self.C_l])

        self.M_u = [None]
        self.M_u_fn = [None]
        self.N_u = [None]
        self.N_u_fn = [None]
        self.Q_u = [None]
        self.Q_u_fn = [None]
        self.S_u = [None]
        self.S_u_fn = [None]
        self.C_u = [None]
        self.C_u_fn = [None]

        self.M_u = jacobian(self.dHu, self.state_var)
        self.M_u_fn = casadi.Function('M_u' + str(self.robot_index),
                                         [self.neighbor_state_var, self.ctrl_var, self.costate,
                                          self.aux_var],
                                         [self.M_u])
        self.N_u = jacobian(self.dHu, self.ctrl_var)
        self.N_u_fn = casadi.Function('N_u' + str(self.robot_index),
                                         [self.neighbor_state_var, self.ctrl_var, self.costate,
                                          self.aux_var],
                                         [self.N_u])

        self.Q_u = [None] * self.num_uav
        self.Q_u_fn = [None] * self.num_uav
        for j in self.neighbors:
            self.Q_u[j] = jacobian(self.dHu, self.state_var[j]) ######
            self.Q_u_fn[j] = casadi.Function('Q_u' + str(self.robot_index) + str(j),
                                                [self.neighbor_state_var, self.ctrl_var, self.costate,
                                                 self.aux_var],
                                                [self.Q_u[j]])
        # Q_u[i][i] = M_u[i]

        self.S_u = jacobian(self.dHu, self.costate)
        self.S_u_fn = casadi.Function('S_u' + str(self.robot_index),
                                         [self.neighbor_state_var, self.ctrl_var, self.costate,
                                          self.aux_var],
                                         [self.S_u])
        self.C_u = jacobian(self.dHu, self.full_aux)
        self.C_u_fn = casadi.Function('C_u' + str(self.robot_index),
                                         [self.neighbor_state_var, self.ctrl_var, self.costate,
                                          self.aux_var],
                                         [self.C_u])

        self.M_x = [None]
        self.M_x_fn = [None]
        self.N_x = [None]
        self.N_x_fn = [None]
        self.Q_x = [None] * self.num_uav
        self.Q_x_fn = [None] * self.num_uav
        self.S_x = [None]
        self.S_x_fn = [None]
        self.C_x = [None]
        self.C_x_fn = [None]

        self.M_x = jacobian(self.dHx, self.state_var)
        self.M_x_fn = casadi.Function('M_x' + str(self.robot_index),
                                      [self.neighbor_state_var, self.ctrl_var, self.costate,
                                       self.aux_var],
                                      [self.M_x])
        self.N_x = jacobian(self.dHx, self.ctrl_var)
        self.N_x_fn = casadi.Function('N_x' + str(self.robot_index),
                                      [self.neighbor_state_var, self.ctrl_var, self.costate,
                                       self.aux_var],
                                      [self.N_x])

        self.Q_u = [None] * self.num_uav
        self.Q_u_fn = [None] * self.num_uav

        for j in self.neighbors:
            self.Q_x[j] = jacobian(self.dHx, self.state_var)
            self.Q_x_fn[j] = casadi.Function('Q_x' + str(self.robot_index) + str(j),
                                             [self.neighbor_state_var, self.ctrl_var, self.costate,
                                              self.aux_var],
                                             [self.Q_x[j]])
        # Q_x[i][i] = M_x[i]

        self.S_x = jacobian(self.dHx, self.costate)
        self.S_x_fn = casadi.Function('S_x' + str(self.robot_index),
                                      [self.neighbor_state_var, self.ctrl_var, self.costate,
                                       self.aux_var],
                                      [self.S_x])
        self.C_x = jacobian(self.dHx, self.full_aux)
        self.C_x_fn = casadi.Function('C_x' + str(self.robot_index),
                                      [self.neighbor_state_var, self.ctrl_var, self.costate,
                                       self.aux_var],
                                      [self.C_x])

        self.MT_x = [None]
        self.MT_x_fn = [None]
        self.CT_x = [None]
        self.CT_x_fn = [None]

        self.MT_x = jacobian(self.dhx, self.state_var)
        self.MT_x_fn = casadi.Function('MT_x' + str(self.robot_index), [self.neighbor_state_var, self.aux_var],
                                          [self.MT_x])

        self.CT_x = jacobian(self.dhx, self.full_aux)
        self.CT_x_fn = casadi.Function('CT_x' + str(self.robot_index), [self.neighbor_state_var, self.aux_var],
                                          [self.CT_x])

    def Compute_state_traj(self, state_traj, ctrl_traj, horizon, auxvar_value):

        time = numpy.array([k for k in range(horizon)])
        self.tau_ctrl_traj = [None]
        self.tau_state_traj = [None]
        self.horizon = [None]
        self.auxvar_value = [None]

        self.horizon = horizon
        self.auxvar_value = auxvar_value
        self.tau_state_traj = np.zeros((self.horizon, self.n_state_var))
        self.tau_ctrl_traj = np.zeros((self.horizon-1, self.n_ctrl_var))
        self.tau_state_traj[0, :] = state_traj
        self.tau_ctrl_traj = ctrl_traj

        # compute state trajectory in \tau from 0 to T-1
        for k in range(self.horizon - 1):
            self.tau_state_traj[k + 1, :] = self.dyn_fn(self.tau_state_traj[k, :], self.tau_ctrl_traj[k, :],
                                                        self.auxvar_value).T #auxvar is \theta in the paper

    def Get_neighbor_state(self,tau_state_traj): # tau_state_traj includes all robots' states
        self.tau_neighbor_state_traj = [None]
        self.tau_neighbor_state_traj = tau_state_traj


    def Compute_Costate_traj(self, loaded_traj=[None], gamma=1e-4, print_level=0,
                             init_option=0, print_gap = 1e2, solver_option='d'):

        self.tau_costate_traj = np.zeros((self.horizon, self.n_state_var))
        self.d_tau_ctrl_traj = np.zeros((self.horizon - 1, self.n_ctrl_var))  # horizon: size of T

        self.tau_costate_traj[-1, :] = self.dhx_fn(self.tau_neighbor_state_traj[self.robot_index][-1, :],
                                                    self.auxvar_value).T
        for k in range(horizon - 2, -1, -1):
            self.tau_costate_traj[k, :] = self.dHx_fn(self.tau_neighbor_state_traj[self.robot_index][k, :],
                                                       self.tau_ctrl_traj[k, :],
                                                       self.tau_costate_traj[k + 1, :], self.auxvar_value).T

        for k in range(horizon - 1):
            self.d_tau_ctrl_traj[k, :] = self.dHu_fn(self.tau_neighbor_state_traj[self.robot_index][k, :],
                                                      self.tau_ctrl_traj[k, :],
                                                      self.tau_costate_traj[k, :], auxvar_value).T

        self.tau_ctrl_traj = self.tau_ctrl_traj - gamma * self.d_tau_ctrl_traj

        # cost_full = 0
        # for k in range(self.horizon - 1):
        #     cost_full = cost_full + self.path_cost_fn(self.tau_neighbor_state_traj[k, :],
        #                                                        self.tau_ctrl_traj[k, :], auxvar_value)
        # cost_full = cost_full + self.final_cost_fn(self.tau_neighbor_state_traj[k, :],
        #                                                     self.auxvar_value)




