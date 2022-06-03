import numpy as np

from planner.rrt_2d import Planner

class Navigator:
    def __init__(self, obstacles, pose, goal, linear_tolerance=0.2, angular_tolerance=5):
        self.pose = [0., 0., 0.]
        self.linear_tolerance = linear_tolerance
        self.angular_tolerance = angular_tolerance * np.pi / 180

        self.milestone = 0
        self.planner = Planner(obstacles)
        self.path = self.planner.plan(pose, goal)

        self.action = [0., 0., 1., 0.]

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
        if self.milestone >= self.path.shape[0]:
            return [0., 0.] + self.action[2:]

        self.pose = pose
        self.action[:2] = self.move_to_target(self.path[self.milestone])
        self.action[2] = self.towards_target(self.path[-1])

        error = (self.pose[0] - self.path[self.milestone][0]) ** 2 + (self.pose[1] - self.path[self.milestone][1]) ** 2
        if error < self.linear_tolerance:
            self.milestone += 1

        return self.action
