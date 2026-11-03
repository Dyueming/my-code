class CliffWalkingEnv:
    """ Cliff walking environment """
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # grid world columns
        self.nrow = nrow  # grid world rows
        # Transition table P[state][action] = [(p, next_state, reward, done)]
        self.P = self.createP()

    def createP(self):
        # Initialisation
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4 actions, change[0]: up, change[1]: down, change[2]: left, change[3]: right. Origin (0,0)
        # Defined at the top-left corner
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # On the cliff or the goal no further interaction is possible, so rewards are 0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # Any other position
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # The next position is the cliff or the goal
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # next position is the cliff
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P