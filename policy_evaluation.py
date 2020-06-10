import numpy as np
import time
import threading
import tkinter

app = tkinter.Tk()
app.title("Policy Evaluation")
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
    MAX_ITER=10000,
    LOOP_MS=0.1,
    PRECISION=3,
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
            if not obs[i][j].blocked and obs[i][j].display != 0:
                print("{0:+.2f}".format(obs[i][j].display), end="\t")
            elif obs[i][j].blocked:
                print("*****", end="\t")
            else:
                print("-----", end="\t")
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

    # Handle goal state differently as treat all the neighbors as starting state ([0][0])
    if i == 5 and j == 5:
        return obs[i][j].reward + cfg["GAMMA"] * obs[0][0].display

    # For every possible actions
    for r, c in valid_neighbors_list:
        if valid(r, c, obs):
            instant_reward = obs[i][j].reward
            future_reward = obs[r][c].display

            if obs[r][c].blocked:  # Hit wall and stay
                future_reward = obs[i][j].display
            res += action_prob * (instant_reward + cfg["GAMMA"] * future_reward)
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


###### GUI methods below ######
def init_btns(arr2d):
    btn_list = []
    for i in range(cfg["ROW"]):
        tmp = []
        for j in range(cfg["COL"]):
            b = tkinter.Button(app, text=arr2d[i][j])
            b.config(height=4, width=8, borderwidth=0)
            b.grid(row=i, column=j)
            tmp.append(b)
        btn_list.append(tmp)

    return btn_list


def update_btn(btn_list, obs, old_obs):
    attr = list(o.display for o in obs.flatten())
    min_val = min(attr)
    for i in range(len(btn_list)):
        for j in range(len(btn_list[i])):
            obs_val = obs[i][j].display

            # Change fg color if changed
            if round(obs[i][j].display, cfg["PRECISION"]) != round(old_obs[i][j].display, cfg["PRECISION"]):
                fg = "black"
            else:
                fg = "grey"

            # Change color according to gradient / walls
            if obs_val > 0:
                bg = (0, 204, 0)
            elif obs_val != 0:
                bg = (
                    255,
                    255 - int(obs_val / min_val * 255),
                    255 - int(obs_val / min_val * 255),
                )
            else:
                bg = (255, 255, 255)

            if not obs[i][j].blocked:
                btn_list[i][j]["text"] = round(obs[i][j].display, cfg["PRECISION"])
                btn_list[i][j]["bg"] = "#%02x%02x%02x" % bg
                btn_list[i][j]["fg"] = fg
            else:  # draw for block
                btn_list[i][j]["text"] = ""
                btn_list[i][j]["bg"] = "black"


def update_gui_val(btn_list, old_obs, obs):
    for i in range(cfg["MAX_ITER"]):
        print(f"Policy Evaluation #{i}")
        old_obs = clone(obs)
        sweep(obs)
        update_btn(btn_list, obs, old_obs)
        app.update()
        time.sleep(cfg["LOOP_MS"])

        if diff(old_obs, obs) < cfg["CONV"]:
            print(f"Reached convergence after #{i}")
            app.destroy()


if __name__ == "__main__":
    obs = gen_obs()
    old_obs = clone(obs)
    set_rewards(obs)
    set_block(obs)

    btn_list = init_btns(np.zeros((cfg["ROW"], cfg["COL"])))
    update_thread = threading.Thread(
        target=update_gui_val,
        kwargs={"btn_list": btn_list, "old_obs": old_obs, "obs": obs},
    )
    update_thread.daemon = True
    update_thread.start()

    app.mainloop()