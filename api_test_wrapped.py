from WrappedInnerEnv import RobotEnv
import numpy as np
import cv2 as cv
import time

env = RobotEnv(time_scale=1,render=True, confrontation=True)

while True:
    obs = env.reset()
    for i in range(4500):
        key = cv.waitKey(1)
        action = [0,0] # Default
        if key == ord("w"):
            action = [0,2]
        elif key == ord("s"):
            action = [0,-2]
        elif key == ord("a"):
            action = [-2, 0]
        elif key == ord("d"):
            action = [2, 0]
        obs, reward, done, info = env.step(action)
        img = env.render()
        cv.imshow("image",img)
        if done:
            break
