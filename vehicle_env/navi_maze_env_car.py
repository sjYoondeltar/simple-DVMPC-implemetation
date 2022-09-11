import sys

sys.path.append(".")

import numpy as np
import cv2
from vehicle_env.vehicle_model import UNICAR

class NAVI_ENV(object):

    def __init__(self, dT=0.05, x_init=[0.0, 0.0, 0], target_fix=None, u_min=[0, -np.pi/6], u_max=[2, np.pi/6],
                t_max=200, level=1,
                reward_type='cart', obs_list=[[5.0, 5.0, 10.0, 2.0], [5.0, -5.0, 10.0, 2.0]],
                coef_dis=0.2, coef_angle=1.2, dis_collision=1.0, terminal_cond=2.0, collision_panalty=-10, goal_reward=1,
                continue_after_collision=True):

        self.dT = dT

        self.car = UNICAR(x_init=x_init, u_min=u_min, u_max=u_max)
        _ = self.car.init_state()

        self.coef_dis = coef_dis
        self.coef_angle = coef_angle

        self.t_max = t_max
        self.t = 0
        self.t_max_reach = False

        self.level = int(level)
        self.reward_type = reward_type

        self.target_p_range = [6, 16]
        self.target_phi_range = [-np.pi/6, np.pi/6]

        self.terminal = False
        self.terminal_cond = terminal_cond
        self.reach = False

        self.target_fix = target_fix

        self.dis_collision = dis_collision

        if obs_list is not None:

            self.obs_pts = np.array([[
                [obs[0]-obs[2]/2, obs[1]-obs[3]/2],
                [obs[0]+obs[2]/2, obs[1]-obs[3]/2],
                [obs[0]+obs[2]/2, obs[1]+obs[3]/2],
                [obs[0]-obs[2]/2, obs[1]+obs[3]/2],
                ] for obs in obs_list])

        else:

            self.obs_pts = None
        
        self.bound_pts = np.array([
            [self.car.pos_min[0], self.car.pos_min[1]],
            [self.car.pos_max[0], self.car.pos_min[1]],
            [self.car.pos_max[0], self.car.pos_max[1]],
            [self.car.pos_min[0], self.car.pos_max[1]]
        ])
        
        self.RT_g = np.matmul(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 20], [0, 0, 0, 1]]), np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]))

        self.RT_b = np.linalg.inv(self.RT_g)

        self.Pmtx = np.array([[200.0, 0, 200.0], [0, 200.0, 200.0]])

        self.trajectory_array = None
        
        self.collision_panalty = collision_panalty
        self.goal_reward = goal_reward
        self.continue_after_collision = continue_after_collision

    def renew_obs_list(self, obs_list):

        if obs_list is not None:

            self.obs_pts = np.array([[
                [obs[0]-obs[2]/2, obs[1]-obs[3]/2],
                [obs[0]+obs[2]/2, obs[1]-obs[3]/2],
                [obs[0]+obs[2]/2, obs[1]+obs[3]/2],
                [obs[0]-obs[2]/2, obs[1]+obs[3]/2],
                ] for obs in obs_list])

        else:

            self.obs_pts = None

    def renew_init(self, x_init):

        self.car.x_init = np.array(x_init, dtype=np.float32).reshape([-1,1])

        _ = self.car.init_state()

    def renew_target(self, target):

        self.target = np.array(target).reshape([-1, 1])

        self.target_fix = self.target.copy()

    def generate_target(self):

        if self.target_fix is not None:

            self.target = np.array(self.target_fix).reshape([-1, 1])

        else:

            if self.level==1:

                target_x = 17.0
                target_y = 16.0 * np.random.randint(-1,2)

                self.target = np.array([target_x, target_y]).reshape([-1, 1])

            else:

                target_contact = True

                while target_contact:

                    target_p = self.target_p_range[0] + np.random.rand(1)*(self.target_p_range[1]-self.target_p_range[0])
                    target_phi = self.target_phi_range[0] + np.random.rand(1)*(self.target_phi_range[1]-self.target_phi_range[0])

                    target_x = target_p*np.cos(target_phi)
                    target_y = target_p*np.sin(target_phi)

                    self.target = np.array([target_x, target_y]).reshape([-1, 1])

                    if self.obs_pts is not None:

                        for ob_idx in range(self.obs_pts.shape[0]):

                            obs_centor = (self.obs_pts[ob_idx, 0, :] + self.obs_pts[ob_idx, 2, :])/2

                            obs_wh = np.abs(self.obs_pts[ob_idx, 0, :] - self.obs_pts[ob_idx, 2, :])

                            target_contact = self.check_collision_rect_obs(self.target[0,:], self.target[1,:], obs_centor[0], obs_centor[1], obs_wh[0], obs_wh[1])

                    else:
                        
                        target_contact = False


    def calc_r(self):
        
        diff = self.target - self.car.x[:2, :]

        self.dist = np.linalg.norm(diff)

        x_rel = diff[0, :]*np.cos(-self.car.x[2,:]) - diff[1, :]*np.sin(-self.car.x[2,:])
        
        y_rel = diff[0, :]*np.sin(-self.car.x[2,:]) + diff[1, :]*np.cos(-self.car.x[2,:])

        diff_angle = np.arctan2(y_rel, x_rel)

        r_reach = self.goal_reward if self.dist < self.terminal_cond else 0

        if self.wall_contact:

            r_wall = self.collision_panalty

        else:

            r_wall = 0.

        if self.obs_contact:

            r_obs = self.collision_panalty

        else:

            r_obs = 0.

        if self.reward_type == 'cart':

            r_target = 0*self.coef_dis + self.coef_dis*(-np.square(self.dist/(self.car.pos_max[0])))

        elif self.reward_type == 'polar':

            r_target = self.coef_dis * (1 - np.square(self.dist/(self.car.pos_max[0]))) + self.coef_angle*(1-np.square(diff_angle/np.pi)[0])

            # r_target = self.coef_dis * ( -np.square(self.dist/(self.car.pos_max[0]))) + self.coef_angle*( -np.square(diff_angle/np.pi)[0])

        else:
            r_target = 0

        if self.dist < self.terminal:

            return r_reach

        elif self.obs_contact:

            return r_obs

        elif self.wall_contact:

            return r_wall

        else:

            return r_target


    def step(self, u):

        xn = self.car.step(u)

        self.check_contact()

        r = self.calc_r()

        self.get_terminal()

        self.t += 1

        if self.t < self.t_max:

            self.t_max_reach = False
        
        else:

            self.t_max_reach = True

        return xn, r, self.terminal


    def project_vec(self, vec):

        vec_aug = np.concatenate([vec[:2, :], np.zeros((1, vec.shape[1])), np.ones((1, vec.shape[1]))], axis=0)

        vec_aug = np.matmul(self.RT_b, vec_aug)

        vec_proj = np.array([vec_aug[0,:]/(vec_aug[2,:]+0.0001), vec_aug[1,:]/(vec_aug[2,:]+0.0001), np.ones((vec.shape[1],))], dtype=np.float32).reshape([3,-1])

        vec_proj = np.matmul(self.Pmtx, vec_proj)

        return vec_proj[:2, :].astype(np.int32)


    def render(self):

        self.img = np.zeros((400, 400, 3), dtype=np.uint8)

        t_pxl = self.project_vec(self.target)

        x_pxl = self.project_vec(self.car.x)

        x_arrow = np.array([self.car.x[0, :] + 2*np.cos(self.car.x[2, :]), self.car.x[1, :] + 2*np.sin(self.car.x[2, :])]).reshape([-1, 1])

        xa_pxl = self.project_vec(x_arrow)

        self.img = cv2.circle(self.img, (t_pxl[0][0], t_pxl[1][0]), 10, (0, 255, 0), 2)

        self.img = cv2.circle(self.img, (x_pxl[0][0], x_pxl[1][0]), 10, (255, 0, 0), 2)

        if self.trajectory_array is not None:

            traj_pxl = self.project_vec(self.trajectory_array[:, :, :2].reshape([-1, 2]).T)

            for traj_xy in zip(traj_pxl[0], traj_pxl[1]):

                self.img = cv2.circle(self.img, traj_xy, 1, (0, 0, 255), 2)

        self.img = cv2.arrowedLine(self.img, (x_pxl[0][0], x_pxl[1][0]), (xa_pxl[0][0], xa_pxl[1][0]), (0, 255, 255), 2)

        if self.obs_pts is not None:

            for ob_idx in range(self.obs_pts.shape[0]):

                obs_pxl = self.project_vec(self.obs_pts[ob_idx, :, :].T)

                self.img = cv2.rectangle(self.img, (np.min(obs_pxl[0,:]), np.min(obs_pxl[1,:])), (np.max(obs_pxl[0,:]), np.max(obs_pxl[1,:])), (255, 0, 255), 2) 

        cv2.imshow("env", self.img)

        cv2.waitKey(10)


    def reset(self):

        _ = self.car.init_state()

        self.generate_target()

        self.t = 0

        self.t_max_reach = False

        self.terminal = False

        self.reach = False

        self.wall_contact = False

        self.obs_contact = False

        return self.car.x, self.target

    def get_terminal(self):
        
        if self.dist < self.terminal_cond:
            self.terminal = True
            self.reach = True

        elif self.wall_contact:
            self.terminal = False if self.continue_after_collision else True
            self.reach = False

        elif self.obs_contact:
            self.terminal = False if self.continue_after_collision else True
            self.reach = False

        else:
            self.terminal = False
            self.reach = False

    def check_contact(self):

        #boundary contact check

        if self.car.pos_max[0] - self.car.x[0,:] < self.dis_collision or self.car.x[0,:] - self.car.pos_min[0] < self.dis_collision:
    
            self.wall_contact = True

        elif self.car.pos_max[1] - self.car.x[1,:] < self.dis_collision or self.car.x[1,:] - self.car.pos_min[1] < self.dis_collision:
            
            self.wall_contact = True

        else:
            
            self.wall_contact = False

        if self.obs_pts is not None:

            for ob_idx in range(self.obs_pts.shape[0]):

                obs_centor = (self.obs_pts[ob_idx, 0, :] + self.obs_pts[ob_idx, 2, :])/2

                obs_wh = np.abs(self.obs_pts[ob_idx, 0, :] - self.obs_pts[ob_idx, 2, :])

                self.obs_contact = self.check_collision_rect_obs(self.car.x[0,:], self.car.x[1,:], obs_centor[0], obs_centor[1], obs_wh[0], obs_wh[1])

                if self.obs_contact:

                    break

        else:

            self.obs_contact = False

    
    def check_collision_rect_obs(self, x, y, cx, cy, w, h):
            
        if  x > cx - w/2 - self.dis_collision and x < cx + w/2 + self.dis_collision:

            if y > cy - h/2 - self.dis_collision and y < cy + h/2 + self.dis_collision:

                collision = True

            else:
                
                collision = False

        else:
            collision = False

        return collision

    def push_traj(self, x_pred):

        self.trajectory_array = x_pred

        



if __name__ == '__main__':

    # wall_list =[[-16.0, 8.0, 8.0, 8.0],
    #             [12.0, 8.0, 16.0, 8.0],
    #             [-16.0, -8.0, 8.0, 8.0],
    #             [12.0, -8.0, 16.0, 8.0]]

    wall_list = None
    
    env = NAVI_ENV(x_init=[0.0, 0, 0.0], level=2, reward_type='polar', t_max=500, obs_list=wall_list)


    for eps in range(3):
    
        print(eps)

        done = False

        _, _ = env.reset()

        print(env.target)

        while not env.t_max_reach and not done :

            u = np.array([2*np.random.rand(1), 2*np.pi/3*np.random.rand(1) - np.pi/3]).reshape([-1, 1])

            xn, r, done = env.step(u)

            env.render()

            if env.obs_contact:
                print("collision")
                print(r)

