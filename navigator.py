import numpy as np

from planner.rrt_2d import plan


class Navigator:
    def __init__(self, obstacles, pose, goal,
                 mid_linear_tolerance=0.3, final_linear_tolerance=0.5, angular_tolerance=5,
                 linear_speed=2, angular_speed=2):
        self.pose = pose
        self.mid_linear_tolerance = mid_linear_tolerance**2
        self.final_linear_tolerance = final_linear_tolerance**2
        self.angular_tolerance = angular_tolerance * np.pi / 180

        self.linear_speed = linear_speed
        self.angular_speed = angular_speed

        self.milestone = 0
        self.obstacles = obstacles
        self.goal = goal
        
    def plan(self):
        self.path = plan(self.obstacles, self.pose, self.goal)

        
        if self.path.size > 1:
            self.path = self.path[1:]
            print("[Info] Done planning")
            return True
        else:
            print("[Fatal] Planning failed.")
            return False            

    def tf(self, target):
        delta_x = target[0] - self.pose[0]
        delta_y = target[1] - self.pose[1]
        x_in_robot = delta_x * np.cos(self.pose[2]) + delta_y * np.sin(self.pose[2])
        y_in_robot = -delta_x * np.sin(self.pose[2]) + delta_y * np.cos(self.pose[2])
        return x_in_robot, y_in_robot

    def move_to_target(self, target):
        delta_linear = np.array(self.tf(target))
        direction = delta_linear / np.linalg.norm(delta_linear)
        direction *= self.linear_speed
        return direction.tolist()

    def move_to_target_linear(self, target):
        delta = np.array([target[0] - self.pose[0],target[1] - self.pose[1]])
        direction = delta / np.linalg.norm(delta)
        return direction.tolist()

    def towards_target(self, target):
        x_in_robot, y_in_robot = self.tf(target)
        delta_theta = np.arctan2(y_in_robot, x_in_robot)

        if abs(delta_theta) < self.angular_tolerance:
            return 0
        elif delta_theta > 0:
            return self.angular_speed
        else:
            return -self.angular_speed

    def navigate(self, pose):
        self.pose = pose
        in_final = False
        done = False
        action = [0., 0., 0., 0.]
        action[2] = self.towards_target(self.path[-1])

        if self.milestone >= self.path.shape[0]:
            return action

        if self.milestone == self.path.shape[0] - 1:
            linear_tolerance = self.final_linear_tolerance
            in_final = True
        else:
            linear_tolerance = self.mid_linear_tolerance

        action[:2] = self.move_to_target(self.path[self.milestone])

        error = (self.pose[0] - self.path[self.milestone][0]) ** 2 + (self.pose[1] - self.path[self.milestone][1]) ** 2
        if error < linear_tolerance:
            self.milestone += 1
            done = True

        return action

    def navigate_linear(self, pose):
        self.pose = pose
        action = [0., 0., 0., 0.]

        if self.milestone >= self.path.shape[0]:
            return action

        if self.milestone == self.path.shape[0] - 1:
            linear_tolerance = self.final_linear_tolerance
        else:
            linear_tolerance = self.mid_linear_tolerance

        action[:2] = self.move_to_target_linear(self.path[self.milestone])

        error = (self.pose[0] - self.path[self.milestone][0]) ** 2 + (self.pose[1] - self.path[self.milestone][1]) ** 2
        
        if error < linear_tolerance:
            self.milestone += 1

        return action