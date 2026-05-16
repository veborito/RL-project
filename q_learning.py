import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from snake_env import SnakeEnv

def run(episodes, is_training=True, render=False):
  
  env = SnakeEnv(render_mode='human' if render else None)
  
  if(is_training):
       q = np.zeros([np.pow(2, env.observation_space.shape[0]), env.action_space.n]) # init a 2^11 (each obs is a boolean and there is 11 obs) x 4 array
  else: 
       f = open('q_learning_model.pkl', 'rb')
       q = pickle.load(f)
       f.close()
  
  learning_rate = 0.1
  discount_factor = 0.99
  
  epsilon = 1
  decay = epsilon / (episodes / 2)
  rng = np.random.default_rng()
  
  rewards_per_episodes = np.zeros(episodes)  
  
  for i in tqdm(range(episodes)):
    state = env.reset()[0]
    terminated = False
    truncated = False
    
    while(not terminated and not truncated):
      if is_training and rng.random() < epsilon:
        action = env.action_space.sample()
      else:
        action = np.argmax(q[state])
      
      new_state, reward, terminated, truncated, _ = env.step_bin(action)
  
      if is_training:  
        q[state, action] = q[state, action] + learning_rate * (
          reward + discount_factor * np.max(q[new_state]) - q[state, action]
        )
        print(q[state])
      
      state = new_state
      
    epsilon = max(epsilon - decay, 0)  
        
    if epsilon == 0:
      # print("here")
      learning_rate = 0.0001 # helps stabilize when we are done exploring
    
    if reward == 1:
      rewards_per_episodes[i] = 1
      
  env.close()
  
  sum_rewards = np.zeros(episodes)
  for t in range(episodes):
    sum_rewards[t] = np.sum(rewards_per_episodes[max(0, t-100):(t+1)])
  plt.plot(sum_rewards)
  plt.savefig('q_learning.png')
  
  if is_training:
    f = open('q_learning_model.pkl', 'wb')
    pickle.dump(q, f)
    f.close()
  
if __name__ == '__main__':
  #run(1000)
  run(1, False, True)
