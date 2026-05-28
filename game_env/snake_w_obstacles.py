import pygame
import numpy as np
import random
from enum import Enum


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class SnakeGame:
    def __init__(self, obs="flat", living_cost=False, width=20, height=15, cell_size=20, num_obstacles=4):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.num_obstacles = num_obstacles

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.DARK_GREEN = (0, 128, 0)

        # Pygame setup (only initialized when needed)
        self.screen = None
        self.clock = None
        self.font = None
        
        self.obs = obs
        self.living_cost = living_cost
        
        self.reset()

    def reset(self):
        """Reset the game to initial state"""
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = Direction.RIGHT
        self.obstacles = self._generate_obstacles()
        self.food = self._generate_food()
        self.score = 0
        self.game_over = False
        return self._get_observation()

    def _generate_obstacles(self):
      obstacles = set()
      while len(obstacles) < self.num_obstacles:
        obstacle = (
              random.randint(0, self.width - 1),
              random.randint(0, self.height - 1),
          )
        if obstacle not in self.snake:
            obstacles.add(obstacle)
      return list(obstacles)
    
    
    def _generate_food(self):
        """Generate food at random position not occupied by snake"""
        while True:
            food = (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
            )
            if food not in self.snake and food not in self.obstacles:
                return food
    
    def take_action(self, action=None):
        """
        Take a action in the game
        For RL: action = Direction enum
        For manual: action = None (uses self.direction directly)
        """
        
        if self.game_over:
            return self._get_observation(), 0, True, False, {}

        # Update direction based on action (only for RL mode)
        if action is not None:
            self._update_direction(action)

        # Move snake
        head_x, head_y = self.snake[0]
        prev_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
        
        # Calculate new head position
        if self.direction == Direction.UP:
            new_head = (head_x, head_y - 1)
        elif self.direction == Direction.DOWN:
            new_head = (head_x, head_y + 1)
        elif self.direction == Direction.LEFT:
            new_head = (head_x - 1, head_y)
        elif self.direction == Direction.RIGHT:
            new_head = (head_x + 1, head_y)
        
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
        # Check collisions
        reward = -0.1  if self.living_cost else 0
        terminated = False
        # Wall collision
        if (
            new_head[0] < 0
            or new_head[0] >= self.width
            or new_head[1] < 0
            or new_head[1] >= self.height
        ):
            self.game_over = True
            terminated = True
            reward = -10
            return (
                self._get_observation(),
                reward,
                terminated,
                False,
                {"score": self.score},
            )
            
        if new_head in self.obstacles:
          print(f"head pos: {new_head}, obstacles: {self.obstacles}")
          print("Colliding with stuff")
          self.game_over = True
          terminated = True
          reward = -10
          return (
              self._get_observation(),
              reward,
              terminated,
              False,
              {"score": self.score},
          )

        # Self collision
        if new_head in self.snake:
            self.game_over = True
            terminated = True
            reward = -10
            return (
                self._get_observation(),
                reward,
                terminated,
                False,
                {"score": self.score},
            )

        # Move snake
        self.snake.insert(0, new_head)

        # Check food collision
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self._generate_food()
        else:
            self.snake.pop()  # Remove tail if no food eaten
            if self.living_cost:
              reward = 0.1 if new_dist < prev_dist else -0.15 # claude idea
        return self._get_observation(), reward, terminated, False, {"score": self.score}

    def _update_direction(self, action):
        """Update direction based on action"""
        if action == 0:  # Continue straight
            return

        directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        current_idx = directions.index(self.direction)

        if action == 1:  # Turn right
            self.direction = directions[(current_idx + 1) % 4]
        elif action == 2:  # Turn left
            self.direction = directions[(current_idx - 1) % 4]

    def _get_observation(self):
        """Get current state observation for RL
            Flat observation space for Q-learning and PPO
        """
        if self.obs == "flat":
          if not self.snake:
              return np.zeros(11, dtype=np.float32)

          head = self.snake[0]

          # Get danger in each direction relative to current direction
          directions = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
          current_dir_idx = directions.index(self.direction)

          # Check danger: straight, right, left
          danger_straight = self._is_collision(head, self.direction)
          danger_right = self._is_collision(head, directions[(current_dir_idx + 1) % 4])
          danger_left = self._is_collision(head, directions[(current_dir_idx - 1) % 4])

          # Direction booleans
          dir_up = self.direction == Direction.UP
          dir_right = self.direction == Direction.RIGHT
          dir_down = self.direction == Direction.DOWN
          dir_left = self.direction == Direction.LEFT

          # Food location relative to head
          food_up = self.food[1] < head[1]
          food_down = self.food[1] > head[1]
          food_left = self.food[0] < head[0]
          food_right = self.food[0] > head[0]

          observation = np.array(
              [
                  danger_straight,
                  danger_right,
                  danger_left,
                  dir_up,
                  dir_right,
                  dir_down,
                  dir_left,
                  food_up,
                  food_down,
                  food_left,
                  food_right,
              ],
              dtype=np.intp,
          )
        
        if self.obs == "game":
          head_marks = {Direction.UP: 5, Direction.RIGHT: 6, Direction.DOWN: 7, Direction.LEFT: 8} # to know in which direction the snake is going
          observation = np.zeros([self.width, self.height])
          #draw snake
          for i, (x, y) in enumerate(self.snake):
            mark = head_marks[self.direction] if i == 0 else 1 
            observation[x, y] = mark
          # Draw food
          observation[self.food[0], self.food[1]] = 2

        return observation

    def _is_collision(self, position, direction):
        """Check if moving in direction from position would cause collision"""
        x, y = position

        # Calculate new position
        if direction == Direction.UP:
            new_pos = (x, y - 1)
        elif direction == Direction.DOWN:
            new_pos = (x, y + 1)
        elif direction == Direction.LEFT:
            new_pos = (x - 1, y)
        elif direction == Direction.RIGHT:
            new_pos = (x + 1, y)

        # Check wall collision
        if (
            new_pos[0] < 0
            or new_pos[0] >= self.width
            or new_pos[1] < 0
            or new_pos[1] >= self.height
        ):
            return True
          
        # Check obstacle collision
        if new_pos in self.obstacles:
          return True
        
        # Check body collision
        if new_pos in self.snake:
            return True

        return False

  
    def render(self, mode="human"):
        """Render the game"""
        if mode == "human":
            if self.screen is None:
                pygame.init()
                window_width = self.width * self.cell_size
                window_height = self.height * self.cell_size + 50  # Extra space for UI
                self.screen = pygame.display.set_mode((window_width, window_height))
                pygame.display.set_caption("Snake Game")
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 32)

            # Clear screen
            self.screen.fill(self.BLACK)

            # Draw snake
            for i, (x, y) in enumerate(self.snake):
                color = self.GREEN if i == 0 else self.DARK_GREEN
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.WHITE, rect, 1)
                
            # Draw obstacles
            for i, (x, y) in enumerate(self.obstacles):
                color = self.WHITE
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, color, rect)

            # Draw food
            food_rect = pygame.Rect(
                self.food[0] * self.cell_size,
                self.food[1] * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            pygame.draw.rect(self.screen, self.RED, food_rect)

            # Draw UI
            ui_y = self.height * self.cell_size + 10
            score_text = self.font.render(f"Score: {self.score}", True, self.WHITE)
            self.screen.blit(score_text, (10, ui_y))

            # Game over overlay
            if self.game_over:
                overlay = pygame.Surface(self.screen.get_size())
                overlay.set_alpha(128)
                overlay.fill(self.BLACK)
                self.screen.blit(overlay, (0, 0))

                game_over_text = self.font.render("GAME OVER", True, self.WHITE)
                score_text = self.font.render(
                    f"Final Score: {self.score}", True, self.WHITE
                )
                restart_text = self.font.render(
                    "Press SPACE to restart or ESC to quit", True, self.WHITE
                )

                window_width = self.width * self.cell_size
                window_height = self.height * self.cell_size + 50

                game_over_rect = game_over_text.get_rect(
                    center=(window_width // 2, window_height // 2 - 40)
                )
                score_rect = score_text.get_rect(
                    center=(window_width // 2, window_height // 2)
                )
                restart_rect = restart_text.get_rect(
                    center=(window_width // 2, window_height // 2 + 40)
                )

                self.screen.blit(game_over_text, game_over_rect)
                self.screen.blit(score_text, score_rect)
                self.screen.blit(restart_text, restart_rect)

            pygame.display.flip()
            if self.clock:
                self.clock.tick(30)

    def close(self):
        """Close the rendering window"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def play_manual(self, fps=8):
        """Play the game with manual controls"""
        print("Snake Game - Manual Control")
        print("Arrow Keys: Move | SPACE: Pause | R: Restart | ESC: Quit")
        print("-" * 50)

        if self.screen is None:
            self.render()  # Initialize display

        paused = False
        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if self.game_over:
                        if event.key == pygame.K_SPACE:
                            self.reset()
                            paused = False
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                    else:
                        if (
                            event.key == pygame.K_UP
                            and self.direction != Direction.DOWN
                        ):
                            self.direction = Direction.UP
                        elif (
                            event.key == pygame.K_DOWN
                            and self.direction != Direction.UP
                        ):
                            self.direction = Direction.DOWN
                        elif (
                            event.key == pygame.K_LEFT
                            and self.direction != Direction.RIGHT
                        ):
                            self.direction = Direction.LEFT
                        elif (
                            event.key == pygame.K_RIGHT
                            and self.direction != Direction.LEFT
                        ):
                            self.direction = Direction.RIGHT
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_r:
                            self.reset()
                            paused = False
                        elif event.key == pygame.K_ESCAPE:
                            running = False

            # Update game (if not paused and not game over)
            if not paused and not self.game_over:
                self.take_action()  # No action parameter = manual mode

            # Render
            self.render()

            # Show pause indicator
            if paused and not self.game_over:
                pause_text = self.font.render("PAUSED", True, self.WHITE)
                window_width = self.width * self.cell_size
                pause_rect = pause_text.get_rect(center=(window_width // 2, 25))
                self.screen.blit(pause_text, pause_rect)
                pygame.display.flip()

            self.clock.tick(fps)

        self.close()
        print(f"Game ended. Final score: {self.score}")


def main():
    """Play Snake manually"""
    import argparse

    parser = argparse.ArgumentParser(description="Play Snake manually")
    parser.add_argument("--width", type=int, default=20, help="Game width")
    parser.add_argument("--height", type=int, default=15, help="Game height")
    parser.add_argument("--speed", type=int, default=8, help="Game speed (1-20)")

    args = parser.parse_args()
    args.speed = max(1, min(20, args.speed))

    game = SnakeGame(width=args.width, height=args.height)

    try:
        game.play_manual(fps=args.speed)
    except KeyboardInterrupt:
        game.close()
        print("\nGame interrupted")


if __name__ == "__main__":
    main()
