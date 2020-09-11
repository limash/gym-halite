# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import gym

def show_gym():
    # env = gym.make('CartPole-v0')
    # env = gym.make('MsPacman - v0')
    env = gym.make('gym_halite:halite-v0')
    env.reset()
    for _ in range(10):
        # env.render()
        env.step(env.action_space.sample())  # take a random action
    # env.close()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    show_gym()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
