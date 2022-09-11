import numpy as np

class SimpleSENSOR(object):

    def __init__(self, n_sensor=7, sensor_max=2, range_sensor=[-np.pi, np.pi]):

        self.n_sensor = n_sensor
        self.sensor_max = sensor_max
        self.sensor_angle = np.linspace(range_sensor[0], range_sensor[1], n_sensor)

        self.sensor_info = sensor_max + np.zeros((n_sensor, 3))

    def update_sensors(self, vehicle_info, obstacle_info, boundary_info):

        self.cx, self.cy, self.yaw = vehicle_info.reshape([-1]).tolist()

        self.transform_end_sensors()

        for s_idx in range(self.n_sensor):

            s = self.sensor_info[s_idx, -2:] - np.array([self.cx, self.cy])
            self.sensor_distance_check = [self.sensor_max]
            self.intersections_check = [self.sensor_info[s_idx, -2:]]

            for ob_idx in range(obstacle_info.shape[0]):

                self.check_obs_cast(s, obstacle_info[ob_idx, :, :])

            self.check_bound_cast(s, boundary_info)

            distance = np.min(self.sensor_distance_check)
            distance_index = np.argmin(self.sensor_distance_check)
            self.sensor_info[s_idx, 0] = distance
            self.sensor_info[s_idx, -2:] = self.intersections_check[distance_index]
            

    def transform_end_sensors(self):
        
        xs = self.sensor_max * np.ones((self.n_sensor, )) * np.cos(self.sensor_angle)
        ys = self.sensor_max * np.ones((self.n_sensor, )) * np.sin(self.sensor_angle)
        xys = np.concatenate([xs.reshape([-1,1]), ys.reshape([-1,1])], axis=1)

        xs_rot = xs * np.cos(self.yaw) - ys * np.sin(self.yaw)
        ys_rot = xs * np.sin(self.yaw) + ys * np.cos(self.yaw)

        self.sensor_info[:, 1] = xs_rot + self.cx
        self.sensor_info[:, 2] = ys_rot + self.cy


    def check_obs_cast(self, s, obstacle_info):
        
        for oi in range(obstacle_info.shape[0]):
            p = obstacle_info[oi]
            r = obstacle_info[(oi + 1) % obstacle_info.shape[0]] - obstacle_info[oi]
            if np.cross(r, s) != 0:
                t = np.cross((np.array([self.cx, self.cy]) - p), s) / np.cross(r, s)
                u = np.cross((np.array([self.cx, self.cy]) - p), r) / np.cross(r, s)
                if 0 <= t <= 1 and 0 <= u <= 1:
                    intersection = np.array([self.cx, self.cy]) + u * s
                    self.intersections_check.append(intersection)
                    self.sensor_distance_check.append(np.linalg.norm(u*s))
        
        
    def check_bound_cast(self, s, boundary_info):

        for oi in range(4):
            p = boundary_info[oi]
            r = boundary_info[(oi + 1) % boundary_info.shape[0]] - boundary_info[oi]
            if np.cross(r, s) != 0:  # may collision
                t = np.cross((np.array([self.cx, self.cy]) - p), s) / np.cross(r, s)
                u = np.cross((np.array([self.cx, self.cy]) - p), r) / np.cross(r, s)
                if 0 <= t <= 1 and 0 <= u <= 1:
                    intersection = p + t * r
                    self.intersections_check.append(intersection)
                    self.sensor_distance_check.append(np.linalg.norm(intersection - np.array([self.cx, self.cy])))


class InRANGEVehicleDetector(object):

    def __init__(self, range_max=2, range_angle=[-3*np.pi/4, 3*np.pi/4], relative_obs=False, limit_lateral=5):

        self.range_max = range_max
        self.range_angle = range_angle
        self.relative_obs = relative_obs
        self.limit_lateral = limit_lateral

    def detect_vehicle_in_range(self, ego_state, sv_state, sv_c_pts_list, RT):

        vehicle_mask = np.full((len(sv_c_pts_list), ), False, dtype=bool)

        for idx_sv, sv_c_pts in enumerate(sv_c_pts_list):

            for idx_c in range(sv_c_pts.shape[0]):

                c_xy = sv_c_pts[idx_c,:]

                dis = np.linalg.norm(c_xy-ego_state[:2,:].T, 2)

                dis_y = np.abs(np.matmul(np.linalg.inv(RT), np.concatenate([(c_xy.reshape([-1, 1])), np.array([[1]])]))[1][0])

                if dis <= self.range_max and dis_y < self.limit_lateral:

                    vehicle_mask[idx_sv] = True

                    break

        if self.relative_obs:

            sv_state = sv_state - np.tile(ego_state, (1, sv_state.shape[1]))

            RT0 = np.copy(RT)
            RT0[:2, 2] = 0.

            for i in range(int(sv_state.shape[0]/2)):

                sv_state[2*i:2*(i+1), :] = np.matmul(np.linalg.inv(RT0), np.concatenate([sv_state[2*i:2*(i+1), :], np.ones((1, sv_state.shape[1]))]))[:2, :]

        sv_state[:, ~vehicle_mask] = 0.

        return vehicle_mask,  sv_state


class LAErrEstimator(object):

    def __init__(self, dist_l=10):

        self.dist_l = dist_l

    def measure_yl(self, local_path, RT):

        local_path_aug = np.concatenate([local_path, np.ones((1, local_path.shape[1]))])

        path_rel = np.matmul(np.linalg.inv(RT), local_path_aug)

        pts_x = np.argmin(np.abs(path_rel[0, :] - self.dist_l))

        err_lat = path_rel[1, pts_x]

        err_head = np.arctan2(err_lat - path_rel[1, pts_x-1], path_rel[0, pts_x] - path_rel[0, pts_x-1])

        return err_lat, err_head