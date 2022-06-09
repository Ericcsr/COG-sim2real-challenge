import numpy as np
import gym
from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import itertools
import param
import platform
import collision_geometry as collider

class RobotEnv(gym.Env):
    def __init__(self, time_scale=10, worker_id=1, render=False, confrontation=False, env=CogEnvDecoder):
        self.time_scale = time_scale
        self.worker_id = worker_id

        env_name = param.CONFRONTATION_NAME if confrontation else param.SIM2REAL_NAME
        
        if env !=None:
            self.cog_env = env(env_name=env_name,
                               no_graphics= not render,
                               time_scale=self.time_scale, 
                               worker_id=self.worker_id)
        else:
            self.cog_env = None
        
        self.collision_map = collider.Map()
        state_high = np.ones(61+param.VEC_DIM)
        state_high[:61] *= 100
        state_high[61:] *= 10
        state_low = np.zeros(61+param.VEC_DIM)
        state_high[-1] = 181
        self.observation_space = gym.spaces.Box(state_high, state_low, dtype=np.float32)
        self.cur_obs = None
        self.last_flag = False

        # Action may need remapping
        self.action_bound = np.array([2.5,2.5])
        self.action_space = gym.spaces.Box(np.ones(2), -np.ones(2), dtype=np.float32)
        
    def step(self,action):
        assert self.cog_env != None
        action = self._inner_policy(self.cur_obs, action[0], action[1])
        
        _obs, reward, done, _info = self.cog_env.step(action)
        self.cur_obs = _obs
        reward = self.calc_rewards_from_state(_obs['vector'], done)

        obs = np.zeros(61+param.VEC_DIM)
        vec_state = _obs['vector']
        obs[:61] = _obs['laser']
        obs[61:] = np.array(list(itertools.chain.from_iterable(vec_state[:2]+vec_state[3:5])))
        
        info = _info[0]
        info['judge_result'] = _info[1]
        return obs, reward, done, info

    # action:
    def _omega_policy(self,self_pose, enemy_pose):
        # Compute angle in world coordinate
        angle_between_robots_world = np.arctan2(enemy_pose[1]-self_pose[1], enemy_pose[0]-self_pose[0])
        angle_between_robots_robot = angle_between_robots_world - self._remap_angle(self_pose[2])
        if angle_between_robots_robot > np.pi:
            angle_between_robots_robot = -2 * np.pi + angle_between_robots_robot
        elif angle_between_robots_robot < -np.pi:
            angle_between_robots_robot = 2 * np.pi + angle_between_robots_robot

        # Compute angular velocity from diff 
        # TODO: May be with in a tolerance
        flag = False
        if np.abs(angle_between_robots_robot) > param.OMEGA_LIMIT * param.TS:
            w = np.sign(angle_between_robots_robot) * param.OMEGA_LIMIT
        else:
            w = angle_between_robots_robot / param.TS
            flag = True
        return w, flag

    # obs is vector data
    # Map vx, vy to world coordinate
    # return velocity on robot
    def _map_velocity(self, obs, v):
        theta = self._remap_angle(obs[0][2])
        return self._rotate_vector(v, -theta)

    def _inner_policy(self, obs, vx, vy):
        v = np.array([vx, vy])
        v = self._map_velocity(obs["vector"], v)
        vec_state = obs['vector']
        w, flag = self._omega_policy(vec_state[0], vec_state[3])
        if self.last_flag:
            #shoot = self._check_if_shootable(obs['laser'][30], vec_state[0], vec_state[3])
            shoot = self._check_if_shootable_map(vec_state[0], vec_state[3])
        else:
            shoot = False
        self.last_flag = flag
        return [v[0], v[1], w, shoot]
        
    # Include robot itself's geometry..
    # More accurate version can be done via collision map
    def _check_if_shootable(self,laser_distance, self_pose, enemy_pose):
        gt_distance = np.linalg.norm([self_pose[0]-enemy_pose[0], self_pose[1]-enemy_pose[1]])
        if np.abs(laser_distance-gt_distance) < 0.25:
            return True
        else:
            return False

    def _check_if_shootable_map(self, self_pose, enemy_pose):
        collide = self.collision_map.line_intersect(collider.Line(np.array(self_pose[:2]), np.array(enemy_pose[:2])))
        return not collide

    def _rotate_vector(self, vector, angle):
        return np.array([np.cos(angle)*vector[0]-np.sin(angle)*vector[1],
                         np.sin(angle)*vector[0]+np.cos(angle)*vector[1]])
        
    
    def _remap_angle(self, angle):
        if angle - np.pi > 0:
            return -(2*np.pi - angle)
        else:
            return angle

    def calc_rewards_from_state(self, obs, done):
        reward = 0
        if done:
            reward += (800-obs[4][0])+obs[1][0]
        # Pure pursue the enemy
        reward -= 10 * (obs[0][0] - obs[3][0]) ** 2 + (obs[0][1] - obs[3][1]) ** 2
        reward += -0.1 * obs[-1][1]
        return reward

    def reset(self):
        assert self.cog_env !=None
        _obs = self.cog_env.reset()
        obstacles = self.parse_obs(_obs)
        self.collision_map.add_obstacles(obstacles)
        self.cur_obs = _obs
        obs = np.zeros(61+param.VEC_DIM)
        vec_state = _obs['vector']
        obs[:61] = _obs['laser']
        obs[61:] = np.array(list(itertools.chain.from_iterable(vec_state[:2]+vec_state[3:5])))
        self.last_flag = False
        return obs

    def render(self,mode="human"):
        assert self.cog_env != None
        return self.cog_env.render(mode)

    def parse_obs(self, obs):
        rand_obs = []
        for i in range(5,10):
            rand_obs.append(np.array(obs['vector'][i][:2]))
        return rand_obs

    def add_obstacles_from_obs(self, obs):
        obstacles = self.parse_obs(obs)
        self.collision_map.add_obstacles(obstacles)

    def remove_add_obs(self):
        self.collision_map.remove_rand_obstacles()

    def calc_angle_diff(self, theta, my_pos, en_pos):
        theta = self._remap_angle(theta)
        delta_angle = np.arctan2(en_pos[1]-my_pos[1], en_pos[0]-my_pos[0])
        diff = delta_angle - theta
        return diff

    def close(self):
        assert self.cog_env != None
        self.cog_env.close()