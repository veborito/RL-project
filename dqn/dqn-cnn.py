import torch
import torch.nn as nn
import torch.nn.functional as F
from game_env.snake_env import SnakeEnv


device = torch.device(torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu')

class QNetwork(nn.Module):
  def __init__(self, input_shape, out_actions):
        super().__init__()

        # https://poloclub.github.io/cnn-explainer/
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer_stack = nn.Sequential(
            nn.Flatten(), # flatten inputs into a single vector
            # After flattening the matrix into a vector, pass it to the output layer. To determine the input shape, use the print() statement in forward()
            nn.Linear(in_features=10*2*2, out_features=out_actions)
        )
        
  def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        #print(x.shape)  # Use this to determine input shape of the output layer.
        x = self.layer_stack(x)
        return x

COLOR_PALETTE = torch.tensor([
    [0.1, 0.1, 0.1],  # 0: Empty
    [0.0, 0.8, 0.0],  # 1: Body
    [1.0, 0.0, 0.0],  # 2: Food
    [0.0, 0.0, 0.0],  # 3: Unused
    [0.0, 0.0, 0.0],  # 4: Unused
    [0.0, 0.4, 1.0],  # 5: Head facing UP
    [0.0, 1.0, 1.0],  # 6: Head facing RIGHT
    [1.0, 0.0, 1.0],  # 7: Head facing DOWN
    [1.0, 1.0, 0.0],  # 8: Head facing LEFT
], dtype=torch.float32).to(device)


def state_to_dqn_input(state)->torch.Tensor:
  tensor = torch.from_numpy(state).long()
  # Instant lookup! No loops.
  rgb_image = COLOR_PALETTE[tensor]  # Shape: (x, y, 3)
  cnn_input = rgb_image.permute(2, 0, 1).unsqueeze(0)
  # To see what the image looks like, uncomment out the following lines AND put a break point at the return statement.
  # pic = cnn_input.squeeze()      # input_tensor[batch][channel][row][column] - use squeeze to remove the batch level
  # pic = torch.movedim(pic, 0, 2)    # rearrange from [channel][row][column] to [row][column][channel]
  # plt.imshow(pic)                   # plot the image
  # plt.show()                        # show the image
  return cnn_input


def run(episodes):
  env = SnakeEnv(obs="game", render_mode='human', width=8, height=8)

  # Load learned policy
  policy_network = QNetwork(input_shape=3, out_actions=4).to(device)
  policy_network.load_state_dict(torch.load("snake_dql_cnn.pt"))
  policy_network.eval()    # switch model to evaluation mode

  for _ in range(episodes):
      state = env.reset()[0]  
      for _ in range(10_000):  
          # Select best action   
          with torch.no_grad():
            action_q_values = policy_network(state_to_dqn_input(state).to(device))
            action = torch.argmax(action_q_values.flatten()).detach().cpu().numpy()

          # Execute action
          state, _, terminated, _, _ = env.step(action)
          if terminated:
              break

  env.close()


if __name__=='__main__':
  run(1)
