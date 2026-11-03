import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    """ Bernoulli multi-armed bandit; K is the number of arms """
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)  # win probability of each of the K arms
        # Probability
        self.best_idx = np.argmax(self.probs)  # arm with the highest win probability
        self.best_prob = self.probs[self.best_idx]  # highest win probability
        self.K = K

    def step(self, k):
        # Pulling arm k returns 1 (win) or 0 (loss) with that arm's win probability
        # win)
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


np.random.seed(1)  # fix the seed for reproducibility
K = 100
bandit_10_arm = BernoulliBandit(K)
print("Created a %d-armed Bernoulli bandit" % K)
print("Best arm is #%d with win probability %.4f" %
      (bandit_10_arm.best_idx, bandit_10_arm.best_prob))