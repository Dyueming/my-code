import copy
import Cliff_Walking

class PolicyIteration:
    """ Policy iteration algorithm """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # initial values are 0
        self.pi = [[0.25, 0.25, 0.25, 0.25]
                   for i in range(self.env.ncol * self.env.nrow)]  # uniform random policy
        self.theta = theta  # policy evaluation convergence threshold
        self.gamma = gamma  # discount factor

    def policy_evaluation(self):  # policy evaluation
        cnt = 1  # counter
        while 1:
            max_diff = 0  # initial error
            new_v = [0] * self.env.ncol * self.env.nrow  # initial state values
            for s in range(self.env.ncol * self.env.nrow):  # iterate over all states
                qsa_list = []  # Q(s, a) for every action of state s
                for a in range(4):  # iterate over the actions of each state
                    qsa = 0  # initial action value
                    # For a deterministic model, assignment is enough
                    # p, next_state, r, done = self.env.P[s][a][0]
                    # qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    for res in self.env.P[s][a]:  # iterate for stochastic models
                      p, next_state, r, done = res
                      qsa = p * (r + self.gamma * self.v[next_state] * (1 - done))
                    # Rewards here depend on the next state, so weight them by transition probability
                    qsa_list.append(self.pi[s][a] * qsa)  # Q values of the 4 actions
                new_v[s] = sum(qsa_list)  # V(s) is the sum of Q(s, a)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break  # converged, stop evaluation
            cnt += 1
        print("Policy evaluation finished after %d rounds" % cnt)

    def policy_improvement(self):  # policy improvement
        for s in range(self.env.nrow * self.env.ncol):  # iterate over all states
            qsa_list = []  # value of each action in this state
            for a in range(4):  # iterate over the actions of each state
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)  # max action value of this state
            cntq = qsa_list.count(maxq)  # how many actions tie for the max Q
            # Spread probability evenly over these actions
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("Policy improvement done")
        return self.pi

    def policy_iteration(self):  # policy iteration
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # deep copy for comparison
            new_pi = self.policy_improvement()
            if old_pi == new_pi: break

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("State values:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # Keep 6 characters for aligned output
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("Policy:")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # Special states, e.g. the cliff in cliff walking
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # goal state
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

env = Cliff_Walking.CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])