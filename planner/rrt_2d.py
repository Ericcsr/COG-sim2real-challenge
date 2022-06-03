import numpy as np


from planner.src.rrt.rrt import RRT
from planner.src.search_space.search_space import SearchSpace
from planner.src.utilities.plotting import Plot

Q = np.array([(80, 40)])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 99999 # max number of samples to take before timing out
prc = 0.1  # probability of checking for a connection to goal
X_dimensions = np.array([(0, 8080), (0, 4480)])  # dimensions of Search Space


class Planner:
    def __init__(self, obstacles):
        self.obstacles = obstacles
        self.X = SearchSpace(X_dimensions, obstacles)

    def plan(self, init, goal):
        # create rrt_search
        init = (init[0]*1000, init[1]*1000)
        goal = (goal[0]*1000, goal[1]*1000)
        rrt = RRT(self.X, Q, init, goal, max_samples, r, prc)
        path = rrt.rrt_search()
        if path:
            ## plot
            # plot = Plot("rrt_2d")
            # plot.plot_tree(self.X, rrt.trees)
            # if path is not None:
            #     plot.plot_path(self.X, path)
            # plot.plot_obstacles(self.X, self.obstacles)
            # plot.plot_start(self.X, init)
            # plot.plot_goal(self.X, goal)
            # plot.draw(auto_open=True)

            path = np.array(path) / 1000
            return path
        else:
            return np.empty(0)


