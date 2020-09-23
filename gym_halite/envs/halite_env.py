from abc import ABC

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

    def __init__(self, debug=False):
        self._current_ship = None
        self._is_debug = debug
        # self._episode_ended = False
        self._env = make('halite', debug=True)
        self._trainer = self._env.train([None])

        obs = self._trainer.reset()
        board = hh.Board(obs, self._env.configuration)
        scalar_features_size = get_data_for_ship(board).shape[0]
        halite_map_size = get_halite_map(board).shape
        # if halite map has one layer (2d), we need to add a dimension
        # since, for example, keras conv2d anticipates 3d arrays
        # if len(halite_map_size) == 2:
        #     self._halite_map_size_two = True
        #     halite_map_size = (*halite_map_size, 1)

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Tuple((
            # for the simplest case with only 1 ship and no shipyards and enemies
            # there is only a halite map of size 21x21
            # each cell has no more than 500 halite, rescale it to 0-1
            spaces.Box(low=0, high=1, shape=halite_map_size, dtype=np.float32),
            # (x, y, ship's halite, overall halite, time)
            spaces.Box(low=0, high=1, shape=(scalar_features_size,), dtype=np.float32)
        ))

    def reset(self):
        # returns time_step
        obs = self._trainer.reset()
        board = hh.Board(obs, self._env.configuration)
        self._current_ship = board.ships['0-1']
        if self._is_debug:
            print(board)
        skalar_features = get_data_for_ship(board)
        halite_map = get_halite_map(board)
        # if self._halite_map_size_two:
        #     halite_map = halite_map[:, np.newaxis]
        state = halite_map, skalar_features
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
        skalar_features = get_data_for_ship(next_board)
        halite_map = get_halite_map(next_board)
        # if self._halite_map_size_two:
        #     halite_map = halite_map[:, np.newaxis]
        state = halite_map, skalar_features

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
            reward += ship.halite
        else:
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
    final_reward = reward + no_shipyards_penalty + lost_penalty
    return np.array(final_reward, dtype=np.int32)

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

def get_ship_observation_vector(board, current_ship_id):
    # player_id = board.current_player.id
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

def get_halite_map(board):
    """
    Return a square array of halite map
    """
    board_side = board.configuration.size
    A = np.zeros((board_side, board_side), dtype=np.float32)
    for point, cell in board.cells.items():
        # the maximum amount of halite in a cell is 500
        A[point.x, point.y] = cell.halite/500
    A = A[..., np.newaxis]
    return A

def get_ship_observation_maps(board, current_ship_id):
    """
    Return an array of size (1+2*players_count, board_side, board_side)
    with information about halite, ships, and shipyards
    """
    player_id = board.current_player.id
    board_side = board.configuration.size
    players_count = len(board.players)

    # we need several boards for the all features
    # one board for a halite map
    # one board for a current ship
    # two boards for each player for: ships, shipyards
    # (without the current ship for the current player)
    A = np.zeros((1 + 1 + 2 * players_count, board_side, board_side),
                 dtype=np.float32)

    for point, cell in board.cells.items():
        # A[0, ...] - halite map layer, with rescaling to 0.x
        A[0, point.x, point.y] = cell.halite / 1000
        # A[1, ...] - the current ship position
        if (cell.ship and
                cell.ship.player_id == player_id and
                cell.ship_id == current_ship_id):
            A[1, point.x, point.y] = (cell.ship.halite + 500) / 1000
        # A[2, ...] - player ships map, ship costs 0.5 (500 before rescaling)
        elif cell.ship and cell.ship.player_id == player_id:
            A[2, point.x, point.y] = (cell.ship.halite + 500) / 1000
        # A[3, ...] - player shipyards map
        if cell.shipyard and cell.shipyard.player_id == player_id:
            # shipyard costs 1
            A[3, point.x, point.y] = 1
        # a pairs of maps of each opponent
        for opponent in board.opponents:
            id = 1 + 1 + 2 * opponent.id
            if cell.ship and cell.ship.player_id == opponent.id:
                A[id, point.x, point.y] = (cell.ship.halite + 500) / 1000
            if cell.shipyard and cell.shipyard.player_id == opponent.id:
                A[id + 1, point.x, point.y] = 1
    return A

def get_data_for_ship(board):
    # A = get_ship_observation_vector(board, ship_id)
    # flatten the array
    # B = A.flatten()
    B = np.array([], dtype=np.float32)
    x, y = board.ships['0-1'].position
    # normalize coordinates to be in (0, 1)
    x, y = (x-10)/21, (y-10)/21
    B = np.append(B, x)
    B = np.append(B, y)
    # an upper bound of a ship's current halite is 500 * 0.25 (mining step quota) * 400 (number of steps)
    # 50000 is the upper bound for a ship, it is 224*224 approximately
    # thus, square root a number of halite and divide by 224 to be in (0, 1)
    B = np.append(B, np.sqrt(board.ships['0-1'].halite)/224)

    # add to the data info about the current scores and a time step
    opponents_halite = np.array([])
    for opponent in board.opponents:
        opponents_halite = np.append(opponents_halite, opponent.halite)

    # for example, 4 ships gather 200000 in total, it will be an upper bound, it is 448*448 approximately
    # so to make a total halite from 0 to 1 we can square root a total halite and divide by 448
    C = np.append(B, np.sqrt(board.current_player.halite)/448)
    C = np.append(C, np.sqrt(opponents_halite)/448)
    # the maximum amount of steps is 400
    C = np.append(C, board.step/400)
    return C
