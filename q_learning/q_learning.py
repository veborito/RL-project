import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from game_env.snake_env import SnakeEnv
from pathlib import Path


def run(episodes=1000, episode_len=10_000,is_training=True, render=False, model='q_learning_model'):
  model_path = Path('./q_learning') / (model + '.pkl')
  reward_path = Path('./q_learning') / (model + '.png')
  error_path = Path('./q_learning') / (model + '_error.png')
  
  env = SnakeEnv(render_mode='human' if render else None)
  
  if(is_training):
       q = np.zeros([int(np.pow(2, env.observation_space.shape[0])), env.action_space.n]) # init a 2^11 (each obs is a boolean and there is 11 obs) x 4 array
  else: 
       f = open(model_path, 'rb')
       q = pickle.load(f)
       f.close()
  
  learning_rate = 0.1
  discount_factor = 0.99
  
  epsilon = 1
  decay = 1 / (episodes / 2)
  rng = np.random.default_rng()
  
  rewards_per_episodes = np.zeros(episodes)  
  training_error = np.zeros(episodes)
  for i in tqdm(range(episodes)):
    state = env.reset_bin()[0]
    terminated = False
    truncated = False
    
    for _ in range(episode_len):
      if is_training and rng.random() < epsilon:
        action = env.action_space.sample()
      else:
        action = np.argmax(q[state])
      
      new_state, reward, terminated, truncated, _ = env.step_bin(action)
      rewards_per_episodes[i] += reward
      if is_training:
        future_q_value = np.max(q[new_state])
        target = reward + discount_factor * future_q_value
        temporal_diff = target - q[state, action]
        q[state, action] = q[state, action] + learning_rate * temporal_diff
        
        training_error[i] += temporal_diff
        
      state = new_state
      if (terminated or truncated):
        break
    epsilon = max(epsilon - decay, 0)  
        
    if epsilon == 0:
      # print("here")
      learning_rate = 0.0001 # helps stabilize when we are done exploring
      
  env.close()
  
  plt.title('Cumul rewards per episode')
  plt.xlabel('Episode')
  plt.ylabel('Reward')
  plt.plot(rewards_per_episodes)
  plt.savefig(reward_path)
  plt.figure()
  plt.title('Cumul training error per episode')
  plt.xlabel('Episode')
  plt.ylabel('Error')
  plt.plot(training_error)
  plt.savefig(error_path)
  
  if is_training:
    f = open(model_path, 'wb')
    pickle.dump(q, f)
    f.close()
  
if __name__ == '__main__':
  N_EPISODES = 1_000
  EPISODE_LEN = 10_000
  # run(episodes=N_EPISODES, episode_len=EPISODE_LEN, model='q_learning_model_1k')
  run(5, 1000,False, True, 'q_learning_model_1k')
