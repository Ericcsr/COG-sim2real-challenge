import numpy as np

from planner.rrt_2d import plan


class Navigator:
    def __init__(self, obstacles, pose, goal, mid_linear_tolerance=0.02, final_linear_tolerance=0.5, angular_tolerance=20):
        self.pose = [0., 0., 0.]
        self.mid_linear_tolerance = mid_linear_tolerance
        self.final_linear_tolerance = final_linear_tolerance
        self.angular_tolerance = angular_tolerance * np.pi / 180

        self.milestone = 0
        self.path = plan(obstacles, pose, goal)

        if self.path.size > 0:
            self.path = self.path[1:]
            print("[Info] Done planning")
        else:
            print("[Fatal] Planning faled.")

    def tf(self, target):
        delta_x = target[0] - self.pose[0]
        delta_y = target[1] - self.pose[1]
        x_in_robot = delta_x * np.cos(self.pose[2]) + delta_y * np.sin(self.pose[2])
        y_in_robot = -delta_x * np.sin(self.pose[2]) + delta_y * np.cos(self.pose[2])
        return x_in_robot, y_in_robot

    def move_to_target(self, target, rate=1.5):
        delta_linear = np.array(self.tf(target))
        direction = delta_linear / np.linalg.norm(delta_linear)
        direction *= rate
        return direction.tolist()

    def towards_target(self, target, rate=2):
        x_in_robot, y_in_robot = self.tf(target)
        delta_theta = np.arctan2(y_in_robot, x_in_robot)

        if abs(delta_theta) < self.angular_tolerance:
            return 0
        elif delta_theta > 0:
            return rate
        else:
            return -rate

    def navigate(self, pose):
        self.pose = pose

        action = [0., 0., 0., 0.]
        action[2] = self.towards_target(self.path[-1])

        if self.milestone >= self.path.shape[0]:
            return action

        if self.milestone == self.path.shape[0]-1:
            linear_tolerance = self.final_linear_tolerance
        else:
            linear_tolerance = self.mid_linear_tolerance

        action[:2] = self.move_to_target(self.path[self.milestone])

        error = (self.pose[0] - self.path[self.milestone][0]) ** 2 + (self.pose[1] - self.path[self.milestone][1]) ** 2
        if error < linear_tolerance:
            self.milestone += 1

        return action
