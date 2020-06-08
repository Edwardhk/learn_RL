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
    BLOCK=[
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
    for i in range(8):
        if i == 4:
            continue
        obs[2][i + 1].blocked = True
    for i in range(5):
        obs[i + 3][4].blocked = True


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
    ## Sweep for dynamics of env
    # Store sweep result to buffer
    for i in range(cfg["ROW"]):
        for j in range(cfg["COL"]):
            obs[i][j].buf = get_neighbors_esum(i, j, obs)
    # Sync rewards update
    for i in range(cfg["ROW"]):
        for j in range(cfg["COL"]):
            obs[i][j].display = obs[i][j].buf

# TODO: Set blocks to cfg
# TODO: Check for covergence
if __name__ == "__main__":
    obs = gen_obs()
    set_rewards(obs)
    set_block(obs)

    display(obs)
    for i in range(100):
        print(f"Policy Evaluation Sweep #{i+1}")
        sweep(obs)
        display(obs)
        time.sleep(1)
