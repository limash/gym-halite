import gym
import time


def show_gym(number_of_iterations):
    env = gym.make('gym_halite:halite-v0', debug=False)
    for i in range(number_of_iterations):
        t0 = time.time()
        env.reset()
        for step in range(399):
            state, reward, done, info = env.step(env.action_space.sample())  # take a random action
            if done: break
        t1 = time.time()
        print(f"A number of steps is {step+1}")
        print(f"Time elapsed is {t1-t0}")


if __name__ == '__main__':
    number_of_games = 10
    show_gym(number_of_games)
