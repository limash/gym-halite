import gym
from gym import spaces

from kaggle_environments import make
import kaggle_environments.envs.halite.helpers as hh

import numpy as np

ACTION_NAMES = {0: 'NORTH',
                1: 'SOUTH',
                2: 'WEST',
                3: 'EAST'}


class HaliteEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, debug=False):
        self._current_ship = None
        self._is_debug = debug
        # self._episode_ended = False
        self._env = make('halite', debug=True)
        self._trainer = self._env.train([None])

        obs = self._trainer.reset()
        board = hh.Board(obs, self._env.configuration)
        obs_size = get_data_for_ship(board, '0-1').shape[0]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=50, shape=(obs_size,), dtype=np.float32)

    def reset(self):
        # returns time_step
        obs = self._trainer.reset()
        board = hh.Board(obs, self._env.configuration)
        self._current_ship = board.ships['0-1']
        if self._is_debug:
            print(board)
        state = get_data_for_ship(board, '0-1')
        # state = tf.convert_to_tensor(state, dtype=tf.float32)
        # state = tf.expand_dims(state, 0)
        # self._episode_ended = False
        return state

    def step(self, action):
        # action is np.float32 from -1 to 1
        action_number = digitize_action(action)

        # if self._episode_ended:
        #     # The last action ended the episode. Ignore the current action
        #     # and start a new episode.
        #     return self.reset()

        actions = {}
        try:
            actions['0-1'] = ACTION_NAMES[action_number]
        except KeyError:
            pass

        obs, _, done, info = self._trainer.step(actions)
        next_board = hh.Board(obs, self._env.configuration)
        state = get_data_for_ship(next_board, '0-1')
        # state = tf.convert_to_tensor(state, dtype=tf.float32)
        # state = tf.expand_dims(state, 0)

        # we pass next_board and current_ship to find this ship on the next
        # board and calculate a reward
        reward = get_ship_reward(next_board, self._current_ship)
        # reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        self._current_ship = next_board.ships['0-1']

        if self._is_debug:
            print(next_board)
            try:
                print(f"Action: {ACTION_NAMES[action_number]}")
            except KeyError:
                print("Collection / staying")
            # print(f"halite = {next_board.current_player.halite}")
            try:
                halite_new = next_board.ships['0-1'].halite
            except KeyError:
                halite_new = 'no (it was destroyed of converted)'
            print(f"ship has {halite_new} halite")
            print(f"ship reward is {reward}")

        # if done:
        #     self._episode_ended = True
        return state, reward, done, info


def get_ship_reward(next_board, ship, action=None):
    reward = 0
    no_shipyards_penalty = 0
    lost_penalty = 0
    # if ship is in the shipyard position
    shipyard_position = False
    # first, try to find the ship in the next board
    try:
        next_ship = next_board.ships[ship.id]
        if next_board.current_player.shipyards is not None:
            for shipyard in next_board.current_player.shipyards:
                if next_ship.position == shipyard.position:
                    shipyard_position = True
        if shipyard_position:
            reward += ship.halite / 100  # /2
        else:
            reward += (next_ship.halite - ship.halite) / 100  # /2
    # if there is not the ship
    except KeyError:
        # it is converted
        if action == 'CONVERT':
            pass
            reward += 0
        # or dead
        else:
            pass
            reward -= 0.5  # - ship.halite/2/1000
    finally:
        pass
        # if next_board.current_player.shipyards == None:
        #     no_shipyards_penalty = -0.1

        # penalty if any opponent has more halite
        # diff = np.array([opponent.halite-next_board.current_player.halite
        #                  for opponent in next_board.opponents]).max()
        # if diff > 0:
        #     lost_penalty = (-1 - diff/1000) * next_board.step/400
    final_reward = reward + no_shipyards_penalty + lost_penalty
    return np.array(final_reward, dtype=np.float32)


def digitize_action(action):
    if action < -0.6:
        action_number = 0
    elif -0.6 <= action < -0.2:
        action_number = 1
    elif -0.2 <= action < 0.2:
        action_number = 2
    elif 0.2 <= action < 0.6:
        action_number = 3
    elif 0.6 <= action:
        action_number = 4
    return action_number


def get_ship_observation_vector(board, current_ship_id):
    player_id = board.current_player.id
    board_side = board.configuration.size

    # a vector
    A = np.zeros((board_side * board_side * 4 + 2,), dtype=np.float32)

    # a packs of 4 entries for each cell
    for i, key in enumerate(board.cells):
        cell = board.cells[key]
        i = 4 * i
        # halite - from 0 to 500
        # make it 0 to 0.5
        A[i] = cell.halite / 1000
        if cell.ship is not None:
            # whom this ship belongs to (id) - 0 to 3 (0 is the player)
            # lets skip zero make it negative from -0.1 to -0.4
            A[i + 1] = -(cell.ship.player_id + 1) / 10
            # its halite - from 0 to inf
            # reduce 1000 times
            A[i + 2] = cell.ship.halite / 1000
            if current_ship_id == cell.ship_id:
                current_cell = cell.position
        else:
            # no ship in the current cell = 0
            A[i + 1] = 0
        if cell.shipyard is not None:
            # whom this shipyard belongs to
            # again, make it negative from -1.1 to -1.4
            A[i + 3] = -(cell.shipyard.player_id + 1) / 10 - 1
        else:
            A[i + 3] = 0

    # add position of the current ship
    A[i + 4], A[i + 5] = (current_cell[0] - 10) / 21, (current_cell[1] - 10) / 21
    return A


def get_data_for_ship(board, ship_id):
    A = get_ship_observation_vector(board, ship_id)
    # flatten the array
    B = A.flatten()

    # add to the data info about the current scores and a time step
    opponents_halite = np.array([])
    for opponent in board.opponents:
        opponents_halite = np.append(opponents_halite, opponent.halite)

    C = np.append(B, board.current_player.halite / 1000)
    C = np.append(C, opponents_halite / 1000)
    C = np.append(C, board.step / 400)
    return C
