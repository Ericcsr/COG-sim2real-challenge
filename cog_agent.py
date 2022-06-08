import numpy as np
import copy
from navigator import Navigator
from map.build_obstacles import add_obstacles
from planner.src.search_space.search_space import SearchSpace
from WrappedInnerEnv import RobotEnv
from filter import Filter
import time

X_dimensions = np.array([(0, 8080), (0, 4480)])  # dimensions of Search Space


class Agent:
    def __init__(self, model_path=None):
        self.model_path = model_path
        # you can customize the necessary attributes here

        self.robot_env = RobotEnv(env=None)
        self.navigator = None
        self.current_goal = -1
        self.last_goals = 0
        self.obstacles = np.load("./map/initial_obstacles.npy")
        self.goals_list = None
        self.stage_two_searchspace = None
        self.pose_buffer = []
        self.buffer_length = 20
        self.last_update = time.time()

    def agent_control(self, obs, done, info):
        # The formats of obs, done, info obey the CogEnvDecoder api
        # realize your agent here
        if info is None:
            self.reset()
            filtered_obs = self.estimate(obs, None)
            self.robot_env.add_obstacles_from_obs(obs)
        else:
            filtered_obs = self.estimate(obs, self.last_action)

        if info==None or info[1][3]<5:
            action = self.actor_stage_1(filtered_obs)
        else:
            action = self.actor_stage_2(filtered_obs)
        self.last_action = action
        # return action:[vel_x, vel_y, vel_w, should_shoot]
        return action

    def actor_stage_1(self, obs):
        vector_data = obs['vector']
        self_pose = vector_data[0]
        enemy_pose = vector_data[3]
        if not self.goals_list:
            self.goals_list = [vector_data[i] for i in range(5, 10)]
            stage_two_obstacles = add_obstacles(self.obstacles, self.goals_list)
            self.stage_2_search_space = SearchSpace(X_dimensions, stage_two_obstacles)
        action = self.stage_one(self_pose, enemy_pose, [vector_data[i] for i in range(5, 10)])
        return action

    def stage_one(self, self_pose, enemy_pose, goals):
        # Update self pose buffer
        self.pose_buffer.append(self_pose)
        if len(self.pose_buffer) > self.buffer_length:
            self.pose_buffer.pop(0)

        if goals[-1][-1]:
            print("[Warning] round 1 already finished")
            return [0., 0., 0., 0.]

        if self.current_goal == -1 or goals[self.current_goal][-1]:
            self.current_goal += 1
            goal = goals[self.current_goal]
            del goals[self.current_goal]
            updated_obstacles = add_obstacles(self.obstacles, goals,size=250)
            updated_obstacles = add_obstacles(updated_obstacles, np.array(enemy_pose[:2]).reshape((1,2)), 300)
            stage_1_search_space = SearchSpace(X_dimensions, updated_obstacles)
            print(f"[Info] Planning for target {self.current_goal+1}.")
            # Cooridinate back trace
            cur = len(self.pose_buffer)-1
            for i in range(len(self.pose_buffer)-1, -1, -1):
                if stage_1_search_space.obstacle_free(self.pose_buffer[i][:2]):
                    cur = i
                    break
            # Worst case
            if cur == 0 and stage_1_search_space.obstacle_free(self.pose_buffer[0][:2]) == False:
                for i in range(2 *self.buffer_length):
                    cand_pose = self_pose + np.random.uniform(-0.1,0.1,2)
                    if stage_1_search_space.obstacle_free(self.pose_buffer[0][:2]):
                        self.pose_buffer[0] = cand_pose
                        break
                else:
                    print("Cannot find a collision free pose")
            for i in range(3):
                self.navigator = Navigator(stage_1_search_space, self.pose_buffer[cur], goal)
                result = self.navigator.plan()
                if result == True:
                    break
                else:
                    if cur > 0:
                        cur -= 1
                    else:
                        self.pose_buffer[0] += np.uniform(-0.1,0.1,2)
            else:
                return np.array([0,0,0,0])
                
            #self.navigator = Navigator(stage_1_search_space, self_pose, goal)
        action = self.navigator.navigate(self_pose)
        return action

    def actor_stage_2(self, obs):
        result = False
        if time.time() - self.last_update > 2:
            # Sample a point within R radius of the enemy
            self.stage_2_navigator = Navigator(self.stage_2_search_space, obs['vector'][0], obs['vector'][3])
            result = self.stage_2_navigator.plan()
            self.last_update = time.time()
        if not (self.stage_2_navigator is None) or result:
            v = self.stage_2_navigator.navigate_linear(obs['vector'][0])
        else:
            v = [0,0]
        action = self.robot_env._inner_policy(obs, v[0], v[1])
        return action

    def estimate(self, obs, action=None):
        current_obs = copy.deepcopy(obs)
        if action is None:
            #load dynamical obstacles
            dyn_obstacles = np.vstack([obs['vector'][5][:2],
                                       obs['vector'][6][:2],
                                       obs['vector'][7][:2],
                                       obs['vector'][8][:2],
                                       obs['vector'][9][:2],
                                       obs['vector'][3][:2]]) # Enemy
            self.filter = Filter(init_obs=obs['laser'][::-1],
                                 init_pose=obs['vector'][0],
                                 dyn_obs=dyn_obstacles,
                                 samples=300)
            current_pose = self.filter.current_pose
        else:
            current_pose = self.filter.filter_obs(obs['vector'], np.array(action))
        
        current_obs['vector'][0] = current_pose.tolist()
        #print(current_obs['vector'][0])
        #input()
        return current_obs

    def reset(self):
        self.robot_env.remove_add_obs()
        self.robot_env.last_flag = False
        self.current_goal = -1

    def api_test(self, self_pose):
        test_goal = (4,2)

        ## Obstacle free test
        if self.stage_two_searchspace.obstacle_free(test_goal):  # return True if not inside an obstacle, False otherwise
            ## Planning
            ## `self.stage_two_searchspace` is already initialized in stage one
            navigator = Navigator(self.stage_two_searchspace, self_pose, test_goal)  # Plan a path
            navigator.navigate(self_pose)  # Follow the path
