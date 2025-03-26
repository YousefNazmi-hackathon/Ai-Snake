import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
from snake_env import SnakeEnv  # Import the custom Gym environment

MAX_MEMORY = 600_000
BATCH_SIZE = 5000  # Increased batch size for better training
LR = 0.03

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.env = SnakeEnv()  # Use the Gym environment
        self.model = Linear_QNet(19, 256, self.env.action_space.n)  # Updated input size to 19
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Debugging: Print the shape of the state
        print(f"State shape: {np.array(state).shape}")
        print(f"Next state shape: {np.array(next_state).shape}")
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(0.01, 40 - self.n_games * 0.01)  # Decaying epsilon
        final_move = [0, 0, 0]
        if random.uniform(0, 1) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def train(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0

        while True:
            state = self.env.reset()  # Reset the Gym environment
            done = False
            score = 0

            while not done:
                # Get action
                final_move = self.get_action(state)

                # Perform action in Gym environment
                next_state, reward, done, info = self.env.step(final_move)

                # Reward adjustment
                if info["score"] > score:
                    reward += 10  # Reward for eating food
                elif next_state[-2]:  # Moving closer to food
                    reward += 1
                else:
                    reward -= 0.5  # Penalty for moving farther from food

                # Train short memory
                self.train_short_memory(state, final_move, reward, next_state, done)

                # Remember
                self.remember(state, final_move, reward, next_state, done)

                state = next_state
                score = info["score"]

            # Train long memory, plot result
            self.n_games += 1
            self.train_long_memory()

            if score > record:
                record = score
                self.model.save()

            print('Game', self.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / self.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

# Replace the train function with the Agent's train method
if __name__ == '__main__':
    agent = Agent()
    agent.train()
    agent.train()
    agent = Agent()
    agent.train()
    agent = Agent()
    agent.train()