from q_learning.q_learning import run

if __name__ == '__main__':
  N_EPISODES = 100_000
  EPISODE_LEN = 10_000
  run(episodes=N_EPISODES, episode_len=EPISODE_LEN, model='q_learning_model_100k')
  # run(5, 1000,False, True, 'q_learning_model_100k')
