import numpy as np

redundancy = 150


def add_obstacles(origin, new_obstacle_centers, size=350):
    new_obstacle_centers = np.array(new_obstacle_centers)[:, :2]
    new_obstacle_centers *= 1000
    left_buttom = new_obstacle_centers - size
    top_right = new_obstacle_centers + size
    new_obstacles = np.concatenate((left_buttom, top_right), axis=1)
    updated_obstaclles = np.concatenate((origin, new_obstacles))
    return updated_obstaclles


if __name__ == "__main__":
    obstacles = np.array([[0., 3280., 1000., 3480.],
                          [7080., 1000., 8080., 1200.],
                          [1500., 0., 1700., 1000.],
                          [6380., 3480., 6580., 4480.],
                          [1500., 2140., 2300., 2340.],
                          [5780., 2140., 6580., 2340.],
                          [3540., 3345., 4540., 3545.],
                          [3540., 935., 4540., 1135.]
                          ])

    obstacles[:, (0,1)] -= redundancy
    obstacles[:, (2,3)] += redundancy

    obstacles = add_obstacles(obstacles, np.array([[4.040, 2.240]]), 350)

    np.save("./map/initial_obstacles.npy", obstacles)
