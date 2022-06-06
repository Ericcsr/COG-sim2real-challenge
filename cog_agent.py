import numpy as np

from navigator import Navigator
from map.build_obstacles import add_obstacles


class Agent:
    def __init__(self, model_path=None):
        self.model_path = model_path
        # you can customize the necessary attributes here

        self.navigator = None
        self.current_goal = -1
        self.init_flag = False

        self.obstacles = np.load("./map/initial_obstacles.npy")

    def agent_control(self, obs, done, info):
        # The formats of obs, done, info obey the  CogEnvDecoder api
        # realize your agent here

        filtered_obs = self.filter(obs, self.init_flag)
        if info==None or info[1][3]<5:
            action = self.actor_stage_1(filtered_obs)
        else:
            action = self.actor_stage_2(filtered_obs)
        if self.init_flag == False:
            self.init_flag = True
        # return action:[vel_x, vel_y, vel_w, should_shoot]
        return action

    def actor_stage_1(self, obs):
        vector_data = obs['vector']
        self_pose = vector_data[0]
        enemy_pose = vector_data[3]
        goals_list = [vector_data[i] for i in range(5,10)]
        action = self.stage_one(self_pose, enemy_pose, goals_list)
        return action

    def stage_one(self, self_pose, enemy_pose, goals):
        if goals[-1][-1]:
            print("[Warning] round 1 already finished")
            return [0., 0., 0., 0.]

        if self.current_goal == -1 or goals[self.current_goal][-1]:
            self.current_goal += 1
            goal = goals[self.current_goal]
            del goals[self.current_goal]
            updated_obstacles = add_obstacles(self.obstacles, goals)
            updated_obstacles = add_obstacles(updated_obstacles, np.array(enemy_pose[:2]).reshape((1,2)), 400)
            print(f"[Info] Planning for target {self.current_goal+1}.")
            self.navigator = Navigator(updated_obstacles, self_pose, goal)

        return self.navigator.navigate(self_pose)

    def actor_stage_2(self, obs):
        # --- #
        ## Global init
        # stage_two_obstacles = add_obstacles(self.obstacles, goals_list)

        ## Init for each goal
        # navigator = Navigator(stage_two_obstacles, self_pose, goal)

        ## navigate in every step
        # navigator.navigate(self_pose)
        # --- #

        print("In Stage 2")
        return [0,0,0,0] # TODO: Replace it with my policy

    def filter(self,obs, init_flag):
        return obs # TODO: Using Jiaqi's filter..
