import numpy as np
import math
import itertools

class UNICAR(object):
    
    def __init__(self, dT=0.05, x_init=[0.0, 0.0, 0], u_min=[0, -np.pi/6], u_max=[2, np.pi/6]):
        
        self.x_init= np.array(x_init, dtype=np.float32).reshape([-1,1])

        self.x_dim = 3
        self.u_dim = 2
        
        self.u_min = u_min
        self.u_max = u_max

        self.pos_min = [-20.0, -20.0]
        self.pos_max = [20.0, 20.0]
        
        self.dT = dT

    def init_state(self):

        x_state = self.x_init
        self.x = x_state
        return x_state

    def step(self, u):

        u[0, :] = np.clip(u[0, :], self.u_min[0], self.u_max[0])
        u[1, :] = np.clip(u[1, :], self.u_min[1], self.u_max[1])

        dx0 = u[0, :]*np.cos(self.x[2, :])
        dx1 = u[0, :]*np.sin(self.x[2, :])
        dx2 = u[1, :]

        dx_state = np.concatenate([dx0.reshape([-1, 1]), dx1.reshape([-1, 1]),dx2.reshape([-1, 1])], axis=0)

        self.x = self.x + self.dT*dx_state

        self.x[1,:] = np.clip(self.x[1, :], self.pos_min[1], self.pos_max[1])
        self.x[0,:] = np.clip(self.x[0, :], self.pos_min[0], self.pos_max[0])

        if self.x[2,:] < -np.pi:
            self.x[2,:] = self.x[2,:] + 2*np.pi
        
        elif self.x[2,:] > np.pi:
            self.x[2,:] = self.x[2,:] - 2*np.pi
        
        else:
            pass

        return self.x

    def calc_jacobian(self, x, u):

        F = np.zeros((self.x_dim, self.x_dim), dtype=float)
        G = np.zeros((self.x_dim, self.u_dim), dtype=float)

        F[0,2] = -u[0, :]*np.sin(x[2, :])
        F[1,2] = u[0, :]*np.cos(x[2, :])
                
        G[0,1] = np.cos(x[2, :])
        G[1,1] = np.sin(x[2, :])
        G[2,0] = 1. 

        return F, G

class BicycleCVCAR(object):
    
    def __init__(
        self,
        x_init=[0.0, 0.0, 0],
        u_min=[0, -np.pi/6],
        u_max=[80/3.6, np.pi/6],
        lf=3.,
        lr=3.,
        width=2,
        dT=0.05,
        pos_min=[-10.0, -8.0],
        pos_max=[410.0, 8.0]):
        
        self.lf = lf
        self.lr = lr
        self.L = lf + lr
        self.width = width

        # self.x_init = np.array(x_init, dtype=np.float32).reshape([-1,1])
        self.x_init = x_init

        self.x_dim = 3
        self.u_dim = 2
        
        self.u_min = u_min
        self.u_max = u_max

        self.contact_centor = np.array([[-self.L/3,0],[0,0],[self.L/3,0]]).T

        self.contact_dis = math.sqrt(2)*2

        self.pos_min = pos_min
        self.pos_max = pos_max
        
        self.dT = dT

        self.RT = np.zeros((3,3))

    def init_state(self):

        self.x_init = np.array(self.x_init, dtype=np.float32).reshape([-1,1])
        
        self.x = self.x_init.copy()

        self.update_RT()

        self.contact_centor_g = np.matmul(self.RT, np.concatenate([self.contact_centor, np.array([[1,1,1]])], axis=0))[:2,:].T
        
        return self.x

    def step(self, u):

        u[0, :] = np.clip(u[0, :], self.u_min[0], self.u_max[0])
        u[1, :] = np.clip(u[1, :], self.u_min[1], self.u_max[1])

        # side slip angle
        betas = np.arctan2(self.lr*np.tan(u[1, :]), self.L)

        dx0 = u[0, :]*np.cos(betas + self.x[2, :])
        dx1 = u[0, :]*np.sin(betas + self.x[2, :])
        dx2 = u[0, :]*np.cos(betas)*np.tan(u[1, :])/self.L

        dx_state = np.concatenate([dx0.reshape([-1, 1]), dx1.reshape([-1, 1]),dx2.reshape([-1, 1])], axis=0)

        self.x = self.x + self.dT*dx_state

        self.x[1,:] = np.clip(self.x[1, :], self.pos_min[1], self.pos_max[1])
        self.x[0,:] = np.clip(self.x[0, :], self.pos_min[0], self.pos_max[0])

        if self.x[2,:] < -np.pi:
            self.x[2,:] = self.x[2,:] + 2*np.pi
        
        elif self.x[2,:] > np.pi:
            self.x[2,:] = self.x[2,:] - 2*np.pi
        
        else:
            pass

        self.update_RT()

        self.contact_centor_g = np.matmul(self.RT, np.concatenate([self.contact_centor, np.array([[1,1,1]])], axis=0))[:2,:].T

        return self.x

    def update_RT(self):

        self.Tmtx = np.array([
            [1, 0, self.x[0][0]],
            [0, 1, self.x[1][0]],
            [0, 0, 1]
        ])

        self.Rmtx = np.array([
            [math.cos(self.x[2][0]), -math.sin(self.x[2][0]), 0],
            [math.sin(self.x[2][0]), math.cos(self.x[2][0]), 0],
            [0, 0, 1]
        ])

        self.RT = np.matmul(self.Tmtx, self.Rmtx)

    def check_contact(self, contact_centor_sur):

        vehicle_contact = False

        for e_idx, s_idx in itertools.product([0,1,2], repeat=2):

            dis = np.linalg.norm(self.contact_centor_g[e_idx,:] - contact_centor_sur[s_idx,:], 2)

            if dis <= self.contact_dis:

                vehicle_contact = True

                break

        return vehicle_contact

    def get_contact_centor(self):

        return self.contact_centor_g


class BicycleCAR(object):
    
    def __init__(
        self,
        x_init=[0.0, 0.0, 0, 60/3.6],
        u_min=[-1, -np.pi/6],
        u_max=[1, np.pi/6],
        lf=3.,
        lr=3.,
        width=2,
        dT=0.05,
        state_min=[-10.0, -8.0, -np.pi, 0.],
        state_max=[410.0, 8.0, np.pi, 80/3.6],
        obs_vxy=False):
        
        self.lf = lf
        self.lr = lr
        self.L = lf + lr
        self.width = width

        self.x_init = x_init

        self.x_dim = 4
        self.u_dim = 2
        
        self.u_min = u_min
        self.u_max = u_max

        self.contact_centor = np.array([[-self.L/3,0],[0,0],[self.L/3,0]]).T

        self.contact_dis = math.sqrt(2)*2

        self.state_min = state_min
        self.state_max = state_max
        
        self.dT = dT

        self.RT = np.zeros((3,3))

        self.obs_vxy = obs_vxy

    def init_state(self):

        self.x_init= np.array(self.x_init, dtype=np.float32).reshape([-1,1])

        x_state = self.x_init
        
        self.x = x_state

        self.update_RT()

        self.contact_centor_g = np.matmul(self.RT, np.concatenate([self.contact_centor, np.array([[1,1,1]])], axis=0))[:2,:].T

        self.measure_obs()
        
        return self.x_obs

    def step(self, u):

        u[0, :] = np.clip(u[0, :], self.u_min[0], self.u_max[0])
        u[1, :] = np.clip(u[1, :], self.u_min[1], self.u_max[1])

        # side slip angle
        betas = np.arctan2(self.lr*np.tan(u[1, :]), self.L)

        dx0 = self.x[3, :]*np.cos(betas + self.x[2, :])
        dx1 = self.x[3, :]*np.sin(betas + self.x[2, :])
        dx2 = self.x[3, :]*np.cos(betas)*np.tan(u[1, :])/self.L
        dx3 = u[0, :]

        dx_state = np.concatenate([dx0.reshape([-1, 1]), dx1.reshape([-1, 1]), dx2.reshape([-1, 1]), dx3.reshape([-1, 1])], axis=0)

        self.x = self.x + self.dT*dx_state

        self.x[3,:] = np.clip(self.x[3, :], self.state_min[3], self.state_max[3])
        self.x[1,:] = np.clip(self.x[1, :], self.state_min[1], self.state_max[1])
        self.x[0,:] = np.clip(self.x[0, :], self.state_min[0], self.state_max[0])

        if self.x[2,:] < self.state_min[2]:
            self.x[2,:] = self.x[2,:] + 2*np.pi
        
        elif self.x[2,:] > self.state_max[2]:
            self.x[2,:] = self.x[2,:] - 2*np.pi
        
        else:
            pass

        self.update_RT()

        self.contact_centor_g = np.matmul(self.RT, np.concatenate([self.contact_centor, np.array([[1,1,1]])], axis=0))[:2,:].T

        self.measure_obs()

        return self.x_obs

    def update_RT(self):

        self.Tmtx = np.array([
            [1, 0, self.x[0][0]],
            [0, 1, self.x[1][0]],
            [0, 0, 1]
        ])

        self.Rmtx = np.array([
            [math.cos(self.x[2][0]), -math.sin(self.x[2][0]), 0],
            [math.sin(self.x[2][0]), math.cos(self.x[2][0]), 0],
            [0, 0, 1]
        ])

        self.RT = np.matmul(self.Tmtx, self.Rmtx)

    def check_contact(self, contact_centor_sur):

        vehicle_contact = False

        for e_idx, s_idx in itertools.product([0,1,2], repeat=2):

            dis = np.linalg.norm(self.contact_centor_g[e_idx,:] - contact_centor_sur[s_idx,:], 2)

            if dis <= self.contact_dis:

                vehicle_contact = True

                break

        return vehicle_contact

    def get_contact_centor(self):

        return self.contact_centor_g
        
    def measure_obs(self):

        self.x_obs = np.copy(self.x)

        if self.obs_vxy:
            
            self.x_obs[2][0] = self.x[3][0] * np.cos(self.x[2][0])
            self.x_obs[3][0] = self.x[3][0] * np.sin(self.x[2][0])

        else:

            pass


class TWOWHEELVehicle(object):
    
    def __init__(self, x_init=[0.0, 0.0, 0], u_min=[-1, -1], u_max=[1, 1], radius=0.25, dT=0.05):
        
        self.radius = radius
        
        self.x_init= np.array(x_init, dtype=np.float32).reshape([-1,1])

        self.x_dim = 3
        self.u_dim = 2
        
        self.u_min = u_min
        self.u_max = u_max

        self.pos_min = [-20.0, -20.0]
        self.pos_max = [20.0, 20.0]
        
        self.dT = dT

    def init_state(self):

        x_state = self.x_init
        self.x = x_state
        self.v = 0
        return x_state

    def step(self, u):
        
        u[0, :] = np.clip(u[0, :], self.u_min[0], self.u_max[0])
        u[1, :] = np.clip(u[1, :], self.u_min[1], self.u_max[1])
        
        self.v = (u[0][0] + u[1][0])/2
        w = (u[1][0] - u[0][0])/2/self.radius

        dx0 = self.v*np.cos(self.x[2, :])
        dx1 = self.v*np.sin(self.x[2, :])
        dx2 = w

        dx_state = np.concatenate([dx0.reshape([-1, 1]), dx1.reshape([-1, 1]),dx2.reshape([-1, 1])], axis=0)

        self.x = self.x + self.dT*dx_state

        self.x[1,:] = np.clip(self.x[1, :], self.pos_min[1], self.pos_max[1])
        self.x[0,:] = np.clip(self.x[0, :], self.pos_min[0], self.pos_max[0])

        if self.x[2,:] < -np.pi:
            self.x[2,:] = self.x[2,:] + 2*np.pi
        
        elif self.x[2,:] > np.pi:
            self.x[2,:] = self.x[2,:] - 2*np.pi
        
        else:
            pass

        return self.x
