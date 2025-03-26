import gym
from gym import spaces
import numpy as np
from game import SnakeGameAI

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.game = SnakeGameAI()
        self.action_space = spaces.Discrete(3)  # [straight, right, left]
        self.observation_space = spaces.Box(low=0, high=1, shape=(19,), dtype=np.float32)  # Updated state size

    def reset(self):
        self.game.reset()
        state = self._get_state()
        return state

    def step(self, action):
        reward, done, score = self.game.play_step(action)
        state = self._get_state()
        return state, reward, done, {"score": score}

    def render(self, mode='human'):
        self.game._update_ui()

    def _get_state(self):
        # Ensure the state is a NumPy array with the correct shape
        state = self.game.get_state()
        assert len(state) == 19, f"State vector has {len(state)} features, expected 19."
        return np.array(state, dtype=np.float32)
