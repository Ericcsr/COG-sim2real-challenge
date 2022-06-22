import numpy as np

from planner.src.rrt.rrt import RRT
from planner.src.utilities.plotting import Plot


Q = np.array([(135, 75)])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 5000 # max number of samples to take before timing out
prc = 0.2  # probability of checking for a connection to goal


def plan(X, init, goal):
    init = (init[0]*1000, init[1]*1000)
    goal = (goal[0]*1000, goal[1]*1000)
    rrt = RRT(X, Q, init, goal, max_samples, r, prc)
    path = rrt.rrt_search()
    if path:
        ## plot
        # plot = Plot("rrt_2d")
        # plot.plot_tree(X, rrt.trees)
        # if path is not None:
        #     plot.plot_path(X, path)
        # plot.plot_obstacles(X, obstacles)
        # plot.plot_start(X, init)
        # plot.plot_goal(X, goal)
        # plot.draw(auto_open=True)

        path = np.array(path) / 1000
        return path
    else:
        return np.empty(0)
