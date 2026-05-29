import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game_env.snake_env import SnakeEnv

torch.manual_seed(123)
device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu')
np.random.seed(123)

class QNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


def run(episodes):
        env = SnakeEnv(render_mode='human', width=8, height=8)

        # Load learned policy
        policy_network = QNetwork(n_states=11, n_actions=4, hidden_dim=128).to(device)
        policy_network.load_state_dict(torch.load("snake_dql_sparse.pt"))
        policy_network.eval()    # switch model to evaluation mode

        for _ in range(episodes):
            state = env.reset()[0]  
            for _ in range(1000):  
                # Select best action   
                with torch.no_grad():
                  action_q_values = policy_network(torch.tensor(state, dtype=torch.float32).to(device))
                  action = torch.argmax(action_q_values.flatten()).detach().cpu().numpy()

                # Execute action
                state, _, terminated, _, _ = env.step(action)
                if terminated:
                   break

        env.close()


if __name__=='__main__':
  run(5)
