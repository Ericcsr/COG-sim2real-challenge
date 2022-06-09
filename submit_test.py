from Cogenvdecoder.CogEnvDecoder import CogEnvDecoder
import numpy as np
from cog_agent import Agent

def save_dynamic_obs(obs):
    data=np.zeros((6, 4))
    for i in range(5):
        data[i,:2] = obs['vector'][5+i][:2]
        data[i,:2] -= 0.15
        data[i,2:] = obs['vector'][5+i][:2]
        data[i,2:] += 0.15
    data[5,:2] = obs['vector'][3][:2]
    data[5,:2] -= 0.15
    data[5,2:] = obs['vector'][3][:2]
    data[5,2:] += 0.15
    data *= 1000
    np.save("dyna_obs.npy", data)
    print(data)

env = CogEnvDecoder(env_name="win_v2.1/cog_sim2real_env.exe", no_graphics=False, time_scale=1, worker_id=1) 

num_eval_episodes = 10

eval_agent = Agent(model_path="")

activated_goals_analy = []
time_token_analy = []
attack_damage_analy = []
score_analy = []
for i in range(num_eval_episodes):

    obs = env.reset()
    done = False
    info = None
    bias = np.random.uniform(-0.5, 0.5, 2)
    
    assert(not (obs['laser'] == 0).all())
    print("Ground Truth:",obs['vector'][0])
    np.save("laser.npy" ,obs['laser'])
    np.save("robot_pose.npy", obs['vector'][0][:3])
    save_dynamic_obs(obs)
    while not done:
        obs['vector'][0][0] += bias[0] + (float(np.random.random(1)) * 0.2 - 0.1)
        obs['vector'][0][1] += bias[1] + (float(np.random.random(1)) * 0.2 - 0.1)
        obs['laser'] += np.random.random(61) * 0.1 - 0.05
        action = eval_agent.agent_control(obs=obs, done=done, info=info)
        obs, reward, done, info = env.step(action)

    num_activted_goals = info[1][3]
    activated_goals_analy.append(num_activted_goals)
    time_token = info[1][1]
    time_token_analy.append(time_token)
    attack_damage = info[1][2]
    attack_damage_analy.append(attack_damage)
    score = info[1][0]
    score_analy.append(score)

mean_activated_goal = np.mean(activated_goals_analy)
mean_time_token = np.mean(time_token_analy)
mean_attack_damage = np.mean(attack_damage_analy)
mean_score = np.mean(score_analy)
print("mean activated goal: {}, mean time token: {}, mean attack damage: {}, mean score: {}".format(
    mean_activated_goal, mean_time_token, mean_attack_damage, mean_score))
