# Cliff walking environment
class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        """Initialise the environment: set the number of grid rows and columns"""
        self.nrow = nrow  # rows
        self.ncol = ncol  # columns
        self.x = 0  # agent column (starts at the far left)
        self.y = self.nrow - 1  # agent row (starts at the bottom row)

    def step(self, action):
        """
        Execute the action and update the position
        Input: action (0: up, 1: down, 2: left, 3: right)
        Returns: next state, reward, done
        """
        # Define 4 actions: up, down, left, right as coordinate changes
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        # Move the agent, keeping it inside the grid
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        # Convert the current position to a state
        next_state = self.y * self.ncol + self.x
        # Each step costs a reward of -1
        reward = -1
        done = False
        # Check for the cliff (lower-right region) or the goal
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            # Falling off the cliff gives a negative reward; the goal ends the episode
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):
        """Reset the agent position to the start state"""
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x