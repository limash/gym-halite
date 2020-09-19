import gym

def show_gym():
    # pip install -e .
    env = gym.make('gym_halite:halite-v0')
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())  # take a random action

if __name__ == '__main__':
    show_gym()
