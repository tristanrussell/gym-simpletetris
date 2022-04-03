import numpy as np
import random
import gym
from gym import spaces

# Adapted from the Tetris engine in the TetrisRL project by jaybutera
# https://github.com/jaybutera/tetrisRL

shapes = {
    'T': [(0, 0), (-1, 0), (1, 0), (0, -1)],
    'J': [(0, 0), (-1, 0), (0, -1), (0, -2)],
    'L': [(0, 0), (1, 0), (0, -1), (0, -2)],
    'Z': [(0, 0), (-1, 0), (0, -1), (1, -1)],
    'S': [(0, 0), (-1, -1), (0, -1), (1, 0)],
    'I': [(0, 0), (0, -1), (0, -2), (0, -3)],
    'O': [(0, 0), (0, -1), (-1, 0), (-1, -1)],
}
shape_names = ['T', 'J', 'L', 'Z', 'S', 'I', 'O']


def rotated(shape, cclk=False):
    if cclk:
        return [(-j, i) for i, j in shape]
    else:
        return [(j, -i) for i, j in shape]


def is_occupied(shape, anchor, board):
    for i, j in shape:
        x, y = anchor[0] + i, anchor[1] + j
        if y < 0:
            continue
        if x < 0 or x >= board.shape[0] or y >= board.shape[1] or board[x, y]:
            return True
    return False


def left(shape, anchor, board):
    new_anchor = (anchor[0] - 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def right(shape, anchor, board):
    new_anchor = (anchor[0] + 1, anchor[1])
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def soft_drop(shape, anchor, board):
    new_anchor = (anchor[0], anchor[1] + 1)
    return (shape, anchor) if is_occupied(shape, new_anchor, board) else (shape, new_anchor)


def hard_drop(shape, anchor, board):
    while True:
        _, anchor_new = soft_drop(shape, anchor, board)
        if anchor_new == anchor:
            return shape, anchor_new
        anchor = anchor_new


def rotate_left(shape, anchor, board):
    new_shape = rotated(shape, cclk=False)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)


def rotate_right(shape, anchor, board):
    new_shape = rotated(shape, cclk=True)
    return (shape, anchor) if is_occupied(new_shape, anchor, board) else (new_shape, anchor)


def idle(shape, anchor, board):
    return (shape, anchor)


def convert_grayscale(board, size):
    border_shade = 0
    background_shade = 128
    piece_shade = 190

    arr = np.array(board, dtype=np.uint8)
    arr = np.transpose(arr)

    shape = arr.shape
    limiting_dim = max(shape[0], shape[1])

    gap_size = (size // 100) + 1
    block_size = ((size - (2 * gap_size)) // limiting_dim) - gap_size

    inner_width = gap_size + (block_size + gap_size) * shape[0]
    inner_height = gap_size + (block_size + gap_size) * shape[1]

    padding_width = (size - inner_width) // 2
    padding_height = (size - inner_height) // 2

    arr[arr == 0] = background_shade
    arr[arr == 1] = piece_shade

    arr = np.repeat(arr, block_size, axis=0)
    arr = np.repeat(arr, block_size, axis=1)

    arr = np.insert(arr,
                    np.repeat([block_size * x for x in range(shape[0] + 1)], [gap_size for _ in range(shape[0] + 1)]),
                    background_shade, axis=0)
    arr = np.insert(arr,
                    np.repeat([block_size * x for x in range(shape[1] + 1)], [gap_size for _ in range(shape[1] + 1)]),
                    background_shade, axis=1)

    arr = np.insert(arr, np.repeat([0, len(arr)], [padding_width, size - (padding_width + len(arr))]), border_shade,
                    axis=0)
    arr = np.insert(arr, np.repeat([0, len(arr[0])], [padding_height, size - (padding_height + len(arr[0]))]),
                    border_shade, axis=1)

    return arr


def convert_grayscale_rgb(array):
    shape = array.shape
    shape = (shape[0], shape[1])
    grayscale = np.reshape(array, newshape=(*shape, 1))

    return np.repeat(grayscale, 3, axis=2)


class TetrisEngine:
    def __init__(self,
                 width,
                 height,
                 lock_delay=0,
                 reward_step=False,
                 penalise_height=False,
                 penalise_height_increase=False,
                 advanced_clears=False,
                 high_scoring=False,
                 penalise_holes=False):
        self.width = width
        self.height = height
        self.board = np.zeros(shape=(width, height), dtype=np.float)
        self.scoring = {
            'reward_step': reward_step,
            'penalise_height': penalise_height,
            'penalise_height_increase': penalise_height_increase,
            'advanced_clears': advanced_clears,
            'high_scoring': high_scoring,
            'penalise_holes': penalise_holes
        }

        # actions are triggered by letters
        self.value_action_map = {
            0: left,
            1: right,
            2: hard_drop,
            3: soft_drop,
            4: rotate_left,
            5: rotate_right,
            6: idle,
        }
        self.action_value_map = dict([(j, i) for i, j in self.value_action_map.items()])
        self.nb_actions = len(self.value_action_map)

        # for running the engine
        self.time = -1
        self.score = -1
        self.holes = 0
        self.lines_cleared = 0
        self.piece_height = 0
        self.anchor = None
        self.shape = None
        self.shape_name = None
        self.n_deaths = 0

        self._lock_delay_fn = lambda x: (x + 1) % (max(lock_delay, 0) + 1)
        self._lock_delay = 0

        # used for generating shapes
        # self.shape_counts = [0] * len(shapes)
        self.shape_counts = dict(zip(shape_names, [0] * len(shapes)))

    def _choose_shape(self):
        values = list(self.shape_counts.values())
        maxm = max(values)
        m = [5 + maxm - x for x in values]
        r = random.randint(1, sum(m))
        for i, n in enumerate(m):
            r -= n
            if r <= 0:
                return shape_names[i]

    def _new_piece(self):
        # Place randomly on x-axis with 2 tiles padding
        # x = int((self.width/2+1) * np.random.rand(1,1)[0,0]) + 2
        self.anchor = (self.width / 2, 0)
        # self.anchor = (x, 0)
        self.shape_name = self._choose_shape()
        self.shape_counts[self.shape_name] += 1
        self.shape = shapes[self.shape_name]

    def _has_dropped(self):
        return is_occupied(self.shape, (self.anchor[0], self.anchor[1] + 1), self.board)

    def _clear_lines(self):
        can_clear = [np.all(self.board[:, i]) for i in range(self.height)]
        new_board = np.zeros_like(self.board)
        j = self.height - 1
        for i in range(self.height - 1, -1, -1):
            if not can_clear[i]:
                new_board[:, j] = self.board[:, i]
                j -= 1
        self.lines_cleared += sum(can_clear)
        self.board = new_board

        return sum(can_clear)

    def _count_holes(self):
        self.holes = np.count_nonzero(self.board.cumsum(axis=1) * ~self.board.astype(bool))
        return self.holes

    def valid_action_count(self):
        valid_action_sum = 0

        for value, fn in self.value_action_map.items():
            # If they're equal, it is not a valid action
            if fn(self.shape, self.anchor, self.board) != (self.shape, self.anchor):
                valid_action_sum += 1

        return valid_action_sum

    def get_info(self):
        return {
            'time': self.time,
            'current_piece': self.shape_name,
            'score': self.score,
            'lines_cleared': self.lines_cleared,
            'holes': self.holes,
            'deaths': self.n_deaths,
            'statistics': self.shape_counts
        }

    def step(self, action):
        self.anchor = (int(self.anchor[0]), int(self.anchor[1]))
        self.shape, self.anchor = self.value_action_map[action](self.shape, self.anchor, self.board)
        # Drop each step
        self.shape, self.anchor = soft_drop(self.shape, self.anchor, self.board)

        # Update time and reward
        self.time += 1
        # reward = self.valid_action_count()
        # reward = random.randint(0, 0)
        reward = 1 if self.scoring.get('reward_step') else 0

        done = False
        if self._has_dropped():
            self._lock_delay = self._lock_delay_fn(self._lock_delay)

            if self._lock_delay == 0:
                self._set_piece(True)
                cleared_lines = self._clear_lines()

                if self.scoring.get('advanced_clears'):
                    scores = [0, 40, 100, 300, 1200]
                    reward += 2.5 * scores[cleared_lines]
                    self.score += scores[cleared_lines]
                elif self.scoring.get('high_scoring'):
                    reward += 1000 * cleared_lines
                    self.score += cleared_lines
                else:
                    reward += 100 * cleared_lines
                    self.score += cleared_lines

                if np.any(self.board[:, 0]):
                    self._count_holes()
                    self.n_deaths += 1
                    done = True
                    reward = -100
                else:
                    self._count_holes()

                    if self.scoring.get('penalise_height'):
                        reward -= sum(np.any(self.board, axis=0))
                    elif self.scoring.get('penalise_height_increase'):
                        new_height = sum(np.any(self.board, axis=0))
                        if new_height > self.piece_height:
                            reward -= 10 * (new_height - self.piece_height)
                        self.piece_height = new_height

                    if self.scoring.get('penalise_holes'):
                        reward -= 5 * self.holes

                    self._new_piece()

        self._set_piece(True)
        state = np.copy(self.board)
        self._set_piece(False)
        return state, reward, done

    def clear(self):
        self.time = 0
        self.score = 0
        self.holes = 0
        self.lines_cleared = 0
        self.piece_height = 0
        self._new_piece()
        self.board = np.zeros_like(self.board)

        return self.board

    def render(self):
        self._set_piece(True)
        state = np.copy(self.board)
        self._set_piece(False)
        return state

    def _set_piece(self, on=False):
        for i, j in self.shape:
            x, y = i + self.anchor[0], j + self.anchor[1]
            if x < self.width and x >= 0 and y < self.height and y >= 0:
                self.board[int(self.anchor[0] + i), int(self.anchor[1] + j)] = on

    def __repr__(self):
        self._set_piece(True)
        s = 'o' + '-' * self.width + 'o\n'
        s += '\n'.join(['|' + ''.join(['X' if j else ' ' for j in i]) + '|' for i in self.board.T])
        s += '\no' + '-' * self.width + 'o'
        self._set_piece(False)
        return s


class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}  # TODO: Add human

    # TODO: Add more reward options e.g. wells
    # TODO: Reorganise on next major release
    def __init__(self,
                 width=10,
                 height=20,
                 obs_type='ram',
                 extend_dims=False,
                 render_mode='rgb_array',
                 reward_step=False,
                 penalise_height=False,
                 penalise_height_increase=False,
                 advanced_clears=False,
                 high_scoring=False,
                 penalise_holes=False,
                 lock_delay=0):
        self.width = width
        self.height = height
        self.obs_type = obs_type
        self.extend_dims = extend_dims
        self.render_mode = render_mode

        self.engine = TetrisEngine(width,
                                   height,
                                   lock_delay,
                                   reward_step,
                                   penalise_height,
                                   penalise_height_increase,
                                   advanced_clears,
                                   high_scoring,
                                   penalise_holes)

        self.action_space = spaces.Discrete(7)

        if obs_type == 'ram':
            if extend_dims:
                self.observation_space = spaces.Box(0, 1, shape=(width, height, 1), dtype=np.float32)
            else:
                self.observation_space = spaces.Box(0, 1, shape=(width, height), dtype=np.float32)
        elif obs_type == 'grayscale':
            if extend_dims:
                self.observation_space = spaces.Box(0, 1, shape=(84, 84, 1), dtype=np.float32)
            else:
                self.observation_space = spaces.Box(0, 1, shape=(84, 84), dtype=np.float32)
        elif obs_type == 'rgb':
            self.observation_space = spaces.Box(0, 1, shape=(84, 84, 3), dtype=np.float32)

    def _get_info(self):
        return self.engine.get_info()

    def step(self, action):
        state, reward, done = self.engine.step(action)
        state = self._observation(state=state)
        state = np.array(state, dtype=np.float32)

        info = self._get_info()
        return state, reward, done, info

    def reset(self, return_info=False):
        state = self.engine.clear()
        state = self._observation(state=state)
        state = np.array(state, dtype=np.float32)

        info = self._get_info()
        return (state, info) if return_info else state

    def _observation(self, mode=None, state=None, extend_dims=None):
        obs = state

        if obs is None:
            obs = self.engine.render()

        new_mode = self.obs_type if mode is None else mode

        if new_mode == 'ram':
            extend = self.extend_dims if extend_dims is None else extend_dims

            return np.reshape(obs, newshape=(self.width, self.height, 1)) if extend else obs
        else:
            obs = convert_grayscale(obs, 84)

            if new_mode == 'grayscale':
                extend = self.extend_dims if extend_dims is None else extend_dims

                return np.reshape(obs, newshape=(84, 84, 1)) if extend else obs
            else:
                return convert_grayscale_rgb(obs)

    def render(self, mode=None):
        new_mode = self.render_mode if mode is None else mode

        if new_mode not in self.metadata.get('render.modes'):
            print("Invalid render mode.")
            return

        obs = self.engine.render()
        obs = convert_grayscale(obs, 160)
        obs = convert_grayscale_rgb(obs)

        if new_mode == 'rgb_array':
            return obs
        else:
            # TODO: Human mode
            return

    def close(self):
        del self.engine
