import numpy as np

from navigator import Navigator
from planner.rrt_2d import Planner
from map.build_obstacles import add_obstacles


class Agent:
    def __init__(self, model_path=None):
        self.model_path = model_path
        # you can customize the necessary attributes here

        self.navigator = None
        self.current_goal = -1

        self.obstacles = np.load("./map/initial_obstacles.npy")

    def agent_control(self, obs, done, info):
        # The formats of obs, done, info obey the  CogEnvDecoder api
        # realize your agent here
        vector_data = obs["vector"]
        self_pose = vector_data[0]
        enemy_pose = vector_data[3]
        goals_list = [vector_data[i] for i in range(5, 10)]

        action = self.round_one(self_pose, enemy_pose, goals_list)
        # return action:[vel_x, vel_y, vel_w, shoud_shoot]
        return action

    def round_one(self, self_pose, enemy_pose, goals):
        if goals[-1][-1]:
            print("[Warning] round 1 already finished")
            return [0., 0., 0., 0.]

        if self.current_goal == -1 or goals[self.current_goal][-1]:
            self.current_goal += 1
            goal = goals[self.current_goal]
            del goals[self.current_goal]
            updated_obstacles = add_obstacles(self.obstacles, np.array(goals)[:, :2])
            updated_obstacles = add_obstacles(updated_obstacles, np.array(enemy_pose[:2]).reshape((1,2)), 350)
            print(f"[Info] Planning for target {self.current_goal+1}.")
            self.navigator = Navigator(updated_obstacles, self_pose, goal)

        return self.navigator.navigate(self_pose)
