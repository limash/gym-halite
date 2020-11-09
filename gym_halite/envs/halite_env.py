from abc import ABC
from collections import OrderedDict

import gym
from gym import spaces

from kaggle_environments import make
import kaggle_environments.envs.halite.helpers as hh

import numpy as np

ACTION_NAMES = {0: 'NORTH',
                1: 'SOUTH',
                2: 'WEST',
                3: 'EAST'}


class HaliteEnv(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self, is_action_continuous=False, debug=False):
        self._current_ship = None
        self._is_debug = debug
        self._is_act_continuous = is_action_continuous
        # self._episode_ended = False
        self._board_size = 5
        self._starting_halite = 5000
        self._env = make('halite',
                         configuration={"size": self._board_size,
                                        "startingHalite": self._starting_halite},
                         debug=True)
        self._trainer = self._env.train([None])
        obs = self._trainer.reset()

        board = hh.Board(obs, self._env.configuration)
        scalar_features_size = get_scalar_features(board).shape
        feature_maps_size = get_feature_maps(board).shape

        # four sides for movements plus idleness
        if self._is_act_continuous:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Dict(OrderedDict(
            # for the simplest case with only 1 ship and no shipyards and enemies
            # there is only a halite map of size 21x21
            # each cell has no more than 500 halite, rescale it to 0-1
            {"feature_maps": spaces.Box(low=0, high=1, shape=feature_maps_size, dtype=np.float32),
             # (x, y, ship's halite, overall halite, time)
             "scalar_features": spaces.Box(low=0,
                                           high=1,
                                           shape=scalar_features_size,
                                           dtype=np.float32)}
        ))

    def reset(self):
        # returns time_step
        # let's make a new environment so agent can study new halite location
        self._env = make('halite',
                         configuration={"size": self._board_size,
                                        "startingHalite": self._starting_halite},
                         debug=True)
        self._trainer = self._env.train([None])
        obs = self._trainer.reset()
        board = hh.Board(obs, self._env.configuration)
        self._current_ship = board.ships['0-1']
        if self._is_debug:
            print(board)
        scalar_features = get_scalar_features(board)
        feature_maps = get_feature_maps(board)
        # if self._feature_maps_size_two:
        #     feature_maps = feature_maps[:, np.newaxis]
        # self._episode_ended = False
        return OrderedDict({"feature_maps": feature_maps, "scalar_features": scalar_features})

    def step(self, action):
        if self._is_act_continuous:
            action_number = digitize_action(action)
        else:
            action_number = action

        actions = {}
        try:
            actions['0-1'] = ACTION_NAMES[action_number]
        except KeyError:
            pass

        obs, _, done, info = self._trainer.step(actions)
        next_board = hh.Board(obs, self._env.configuration)
        scalar_features = get_scalar_features(next_board)
        feature_maps = get_feature_maps(next_board)
        state = OrderedDict({"feature_maps": feature_maps, "scalar_features": scalar_features})

        # we pass next_board and current_ship to find this ship on the next
        # board and calculate a reward
        reward = get_ship_reward(next_board, self._current_ship)
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
            print(f"coordinates are {scalar_features}")

        # if done:
        #     self._episode_ended = True
        return state, reward, done, info


def digitize_action(action):
    if action < -0.6:
        action_number = 0  # north
    elif -0.6 <= action < -0.2:
        action_number = 1  # south
    elif -0.2 <= action < 0.2:
        action_number = 2  # west
    elif 0.2 <= action < 0.6:
        action_number = 3  # east
    elif 0.6 <= action:
        action_number = 4  # stay
    return action_number


def get_scalar_features(board):
    A = np.array([], dtype=np.float32)
    x, y = board.ships['0-1'].position

    def to_binary(m, d):
        # m is a number of positions in a binary number
        # d is an array of numbers to convert to binary
        reversed_order = ((d[:, None] & (1 << np.arange(m))) > 0).astype(int)
        return np.fliplr(reversed_order)

    # transform coordinates to binary arrays
    m = 5  # the number of positions in a binary number,
    # 5 is enough for up to 31 decimal, 32 is 2^5
    d = np.array([x, y])
    result = to_binary(m, d)
    A = np.append(A, result[0, :])
    A = np.append(A, result[1, :])

    # the current ship halite =: (0, 1)
    # an upper bound of a ship's current halite is
    # 500 * 0.25 (mining step quota) * 400 (number of steps)
    # 50000 is the upper bound for a ship, it is 224*224 approximately
    # thus, square root a number of halite and divide by 224 to be in (0, 1)
    A = np.append(A, np.sqrt(board.ships['0-1'].halite) / 224)
    A = np.where(A > 1, 1, A)

    # add to the data info about the current scores and a time step
    # opponents_halite = np.array([])
    # for opponent in board.opponents:
    #     opponents_halite = np.append(opponents_halite, opponent.halite)

    # for example, 4 ships gather 200000 in total, it will be an upper bound,
    # it is 448*448 approximately
    # so to make a total halite from 0 to 1 we can square root a total halite and divide by 448
    # B = np.append(A, np.sqrt(board.current_player.halite) / 448)
    # B = np.append(B, np.sqrt(opponents_halite) / 448)
    # the maximum amount of steps is 400
    # B = np.append(B, board.step / 400)
    return A


def get_halite_map(board):
    """
    Return an array of halite map [1, x, y]
    """
    board_side = board.configuration.size
    A = np.zeros((board_side, board_side), dtype=np.float32)
    for point, cell in board.cells.items():
        # the maximum amount of halite in a cell is 500
        A[point.x, point.y] = cell.halite / 500
    # when initializing there can be more than 500 halite in some cells
    A = np.where(A > 1, 1, A)
    A = A[..., np.newaxis]
    return A


def get_feature_maps(board):
    """
    Return an array of size (3, board_side, board_side)
    with information about halite and ships
    """
    board_side = board.configuration.size

    # we need several boards for the all features
    # one board for a halite map
    # plus two boards for ships and their halite
    # summary, for each cell:
    # [ halite,
    #   is ship,
    #   halite in the ship]
    A = np.zeros((1 + 2, board_side, board_side), dtype=np.float32)

    for point, cell in board.cells.items():
        # A[0, ...] - halite map layer, with rescaling to 0.x
        A[0, point.x, point.y] = cell.halite / 500
        # A[1, ...] - the current ship position
        if cell.ship is not None:
            A[1, point.x, point.y] = 1
            # see get_scalar_features
            A[2, point.x, point.y] = np.sqrt(cell.ship.halite) / 224

    A = np.where(A > 1, 1, A)
    return A


def get_ship_reward(next_board, ship, action=None):
    reward = 0
    no_shipyards_penalty = 0
    lost_penalty = 0
    # if ship is in the shipyard position
    shipyard_position = False
    # first, try to find the ship in the next board
    try:
        next_ship = next_board.ships[ship.id]
        # if next_board.current_player.shipyards is not None:
        #     for shipyard in next_board.current_player.shipyards:
        #         if next_ship.position == shipyard.position:
        #             shipyard_position = True
        # if shipyard_position:
        #     reward += ship.halite
        # else:
        #     reward += (next_ship.halite - ship.halite)
        reward += (next_ship.halite - ship.halite)
    # if there is not the ship
    except KeyError:
        # it is converted
        if action == 'CONVERT':
            pass
            reward += 0
        # or dead
        else:
            pass
            reward -= 0  # - ship.halite/2/1000
    finally:
        pass
        # if next_board.current_player.shipyards == None:
        #     no_shipyards_penalty = -0.1

        # penalty if any opponent has more halite
        # diff = np.array([opponent.halite-next_board.current_player.halite
        #                  for opponent in next_board.opponents]).max()
        # if diff > 0:
        #     lost_penalty = (-1 - diff/1000) * next_board.step/400
    # the maximum posiible reward should be 125 (0.25*500)
    # but during initialization there can be more than 500 halite in a cell
    reward = reward/125
    reward = 1 if reward > 1 else reward
    final_reward = reward + no_shipyards_penalty + lost_penalty
    return np.array(final_reward, dtype=np.float32)
