import numpy as np
import time

cfg = dict(
    ROW=10,
    COL=10,
    GAMMA=0.9,
    ACTION_DIM=4,
    NEG_REWARDS=[
        [3, 3],
        [4, 5],
        [4, 6],
        [5, 6],
        [5, 8],
        [6, 8],
        [7, 3],
        [7, 5],
        [7, 6],
    ],
    POS_REWARDS=[[5, 5]],
    BLOCKS=[
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 6],
        [2, 7],
        [2, 8],
        [3, 4],
        [4, 4],
        [5, 4],
        [6, 4],
        [7, 4],
    ],
    CONV=0.001,
    MAX_ITER=10000
)


class Grid:
    def __init__(self):
        self.blocked = False
        self.display = 0
        self.reward = 0
        self.buf = 0


def display(obs):
    for i in range(len(obs)):
        for j in range(len(obs[i])):
            if not obs[i][j].blocked:
                print("{0:+.2f}".format(obs[i][j].display), end="\t")
            else:
                print("XXXXX", end="\t")
        print()
    print()


def gen_obs():
    obs = np.empty((cfg["ROW"], cfg["COL"]), dtype=object)
    for i in range(cfg["ROW"]):
        for j in range(cfg["COL"]):
            obs[i][j] = Grid()
    return obs


def set_rewards(obs):
    for i, j in cfg["NEG_REWARDS"]:
        obs[i][j].reward = -1
    for i, j in cfg["POS_REWARDS"]:
        obs[i][j].reward = 1


def set_block(obs):
    for i, j in cfg["BLOCKS"]:
        obs[i][j].blocked = True


def valid(i, j, obs):
    if i < 0 or i >= cfg["ROW"]:
        return False
    if j < 0 or j >= cfg["COL"]:
        return False
    return True


def get_valid_neighbors_list(r, c, obs):
    res = []
    neighbor = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    for i in range(cfg["ACTION_DIM"]):
        newr = r + neighbor[i][0]
        newc = c + neighbor[i][1]
        if valid(newr, newc, obs):
            res.append([newr, newc])
    return res


def get_neighbors_esum(i, j, obs):
    res = 0
    valid_neighbors_list = get_valid_neighbors_list(i, j, obs)
    if len(valid_neighbors_list) == 0:
        return 0
    action_prob = 1 / len(valid_neighbors_list)

    # For every possible actions
    for r, c in valid_neighbors_list:
        if valid(r, c, obs):
            future_reward = obs[r][c].display
            if obs[r][c].blocked:
                future_reward = obs[i][j].display
            res += action_prob * (obs[i][j].reward + cfg["GAMMA"] * future_reward)
    return res


def sweep(obs):
    # Store sweep result to buffer
    for i in range(cfg["ROW"]):
        for j in range(cfg["COL"]):
            obs[i][j].buf = get_neighbors_esum(i, j, obs)
    # Sync rewards update
    for i in range(cfg["ROW"]):
        for j in range(cfg["COL"]):
            obs[i][j].display = obs[i][j].buf


def diff(current_obs, previous_obs):
    # for i in np.concatenate(current_obs):
    #     print(i.display)
    res = 0.0
    for i in range(cfg["ROW"]):
        for j in range(cfg["COL"]):
            res += current_obs[i][j].display - previous_obs[i][j].display
    return abs(res)


def clone(obs):
    res = np.empty((cfg["ROW"], cfg["COL"]), dtype=object)
    for i in range(cfg["ROW"]):
        for j in range(cfg["COL"]):
            res[i][j] = Grid()
            res[i][j].blocked = obs[i][j].blocked
            res[i][j].display = obs[i][j].display
            res[i][j].reward = obs[i][j].reward
            res[i][j].buf = obs[i][j].buf
    return res


# TODO: Check for covergence
if __name__ == "__main__":
    obs = gen_obs()
    set_rewards(obs)
    set_block(obs)
    display(obs)

    for i in range(cfg["MAX_ITER"]):
        print(f"Policy Evaluation Sweep #{i+1}")
        old_obs = clone(obs)
        sweep(obs)

        if(diff(old_obs, obs) < cfg["CONV"]):
            print(f"Reached convergence after sweep #{i+1}")
            display(obs)
            break
