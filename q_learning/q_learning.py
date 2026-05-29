import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from game_env.snake_env import SnakeEnv
from pathlib import Path

np.random.seed(123)

def run(episodes=1000, episode_len=10_000, is_training=True, living_cost=False, render=False, model='q_learning_model'):
  model_path = Path('./q_learning') / (model + '.pkl')
  
  if not living_cost:
    env = SnakeEnv(render_mode='human' if render else None, width=8, height=8)
  else:
    env = SnakeEnv(render_mode='human' if render else None, width=8, height=8, living_cost=True)
  
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
  rng = np.random.default_rng(seed=123)
  
  rewards_per_episodes = np.zeros(episodes)  
  training_error = np.zeros(episodes)
  scores = []
  for i in tqdm(range(episodes)):
    state = env.reset_bin()[0]
    terminated = False
    truncated = False
    score = 0
    for _ in range(episode_len):
      if is_training and rng.random() < epsilon:
        action = env.action_space.sample()
      else:
        action = np.argmax(q[state])
      
      new_state, reward, terminated, truncated, info = env.step_bin(action)
      rewards_per_episodes[i] += reward
      if is_training:
        future_q_value = np.max(q[new_state])
        target = reward + discount_factor * future_q_value
        temporal_diff = target - q[state, action]
        q[state, action] = q[state, action] + learning_rate * temporal_diff
        
        training_error[i] += temporal_diff
        
      state = new_state
      score = info["score"]
      if (terminated or truncated):
        break
    epsilon = max(epsilon - decay, 0)  
    scores.append(score)    
    if epsilon == 0:
      # print("here")
      learning_rate = 0.0001 # helps stabilize when we are done exploring
      
  env.close()
  
  if is_training:
    f = open(model_path, 'wb')
    pickle.dump(q, f)
    f.close()

  return rewards_per_episodes, scores, training_error

# running mean function for the purpose of visualization
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


if __name__ == '__main__':
  N_EPISODES = 10_000
  EPISODE_LEN = 1000
  
  model = "q_learning_model_10k"
  
  rewards_per_episodes, scores, training_error = run(episodes=N_EPISODES, episode_len=EPISODE_LEN, model="q_learning_model_10k_sparse")
  rewards_per_episodes_dense, scores_dense, training_error_dense = run(episodes=N_EPISODES, episode_len=EPISODE_LEN, living_cost=True, model="q_learning_model_10k_dense")
  
  plt.title('Cumul rewards per episode')
  plt.xlabel('Episode')
  plt.ylabel('Reward')
  plt.plot(running_mean(rewards_per_episodes, 100), label="sparse")
  plt.plot(running_mean(rewards_per_episodes_dense, 100), label="dense")
  plt.grid()
  plt.legend()
  plt.savefig(Path('./q_learning') / (model + '_rewards.png'))
  plt.figure()
  plt.title('Cumul training error per episode')
  plt.xlabel('Episode')
  plt.ylabel('Error')
  plt.plot(running_mean(training_error, 100), label="sparse")
  plt.plot(running_mean(training_error_dense, 100),  label="dense")
  plt.grid()
  plt.legend()
  plt.savefig(Path('./q_learning') / (model + '_error.png'))
  plt.figure()
  plt.title('Score per episode')
  plt.xlabel('Episode')
  plt.ylabel('Score')
  plt.plot(running_mean(scores, 100), label="sparse")
  plt.plot(running_mean(scores_dense, 100),  label="dense")
  plt.grid()
  plt.legend()
  plt.savefig(Path('./q_learning') / (model + '_score.png'))
  
  #run(episodes=5, episode_len=200, False, True, 'q_learning_model_10k_sparse')
  #run(episodes=5, episode_len=200, False, True, 'q_learning_model_10k_dense')
