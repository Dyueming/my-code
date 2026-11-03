import Cliff_Walking as C_W
from PolicyIteration import print_agent
class ValueIteration:
    """ Value iteration algorithm """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # initial values are 0
        self.theta = theta  # value convergence threshold
        self.gamma = gamma
        # Policy obtained after value iteration
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]

    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []  # Q(s, a) for every action of state s
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)  # this line is the key difference from policy iteration
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break  # converged, stop evaluation
            cnt += 1
        print("Value iteration converged after %d rounds" % cnt)
        self.get_policy()

    def get_policy(self):  # derive a greedy policy from the values
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # how many actions tie for the max Q
            # Spread probability evenly over these actions
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]


env = C_W.CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])