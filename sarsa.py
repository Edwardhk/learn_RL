import numpy as np
import time
import threading
import tkinter

app = tkinter.Tk()
app.title("Policy Evaluation")
cfg = dict(
    ROW=10,
    COL=10,
    ALPHA=0.5,
    GAMMA=0.9,
    EPSILON=0.3,
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
    MAX_ITER=100000,
    LOOP_SECOND=0.01,
    PRECISION=3,
)
currentX = 5
currentY = 5


class Grid:
    def __init__(self):
        self.blocked = False
        self.display = 0
        self.reward = 0
        self.buf = 0


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


def display(obs):
    for i in range(len(obs)):
        for j in range(len(obs[i])):
            if i == currentX and j == currentY:
                print("OOOOO", end="\t")
                continue
            if not obs[i][j].blocked and obs[i][j].display != 0:
                print("{0:+.2f}".format(obs[i][j].display), end="\t")
            elif obs[i][j].blocked:
                print("*****", end="\t")
            else:
                print("-----", end="\t")
        print()
    print()


def sarsa_update(i, j, obs):
    neighbors_list = get_valid_neighbors_list(i, j, obs)
    rand_index = np.random.randint(0, len(neighbors_list) - 1)

    # Adjust current position randomly by epsilon
    if np.random.random(1)[0] <= cfg["EPSILON"]:
        r = np.random.randint(0, cfg["ROW"])
        c = np.random.randint(0, cfg["COL"])
    else:
        r, c = neighbors_list[rand_index]

    # Special policy for grid world
    if i == 5 and j == 5:
        future_reward = obs[0, 0].display
    elif obs[r][c].blocked:
        future_reward = obs[i, j].display
    else:
        future_reward = obs[r, c].display

    # Compute the stepping value update
    obs[i, j].display += cfg["ALPHA"] * (
        obs[i, j].reward + cfg["GAMMA"] * future_reward - obs[i, j].display
    )
    
    if obs[r][c].blocked:
        return i, j
    return r, c

###### GUI methods below ######
def init_btns(arr2d):
    btn_list = []
    for i in range(cfg["ROW"]):
        tmp = []
        for j in range(cfg["COL"]):
            b = tkinter.Label(app, text=arr2d[i][j])
            b.config(height=4, width=8, borderwidth=0)
            b.grid(row=i, column=j)
            tmp.append(b)
        btn_list.append(tmp)

    return btn_list


def update_btn(btn_list, obs):
    attr = list(o.display for o in obs.flatten())
    min_val = min(attr)

    i = currentX
    j = currentY
    obs_val = obs[i][j].display

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

    btn_list[i][j]["text"] = round(obs[i][j].display, cfg["PRECISION"])
    btn_list[i][j]["bg"] = "#%02x%02x%02x" % bg


def update_gui_val(btn_list, obs):
    global currentX 
    global currentY

    # Draw walls
    for i in range(cfg["ROW"]):
        for j in range(cfg["COL"]):
            if(obs[i, j].blocked):
                btn_list[i][j]["text"] = ""
                btn_list[i][j]["bg"] = "black"


    for i in range(cfg["MAX_ITER"]):
        print(f"SARSA steps #{i}")
        currentX, currentY = sarsa_update(currentX, currentY, obs)
        update_btn(btn_list, obs)
        app.update()
    input()
    app.destroy()

if __name__ == "__main__":
    obs = gen_obs()
    set_rewards(obs)
    set_block(obs)

    btn_list = init_btns(np.zeros((cfg["ROW"], cfg["COL"])))
    update_thread = threading.Thread(
        target=update_gui_val,
        kwargs={"btn_list": btn_list, "obs": obs},
    )
    update_thread.daemon = True
    update_thread.start()

    app.mainloop()
