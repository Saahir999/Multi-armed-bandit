import math

import matplotlib.pyplot as plt
import numpy as np


class MRP:
    def __init__(self):
        self.path = ["Lend", "A", "B", "C", "D", "E", "Rend"]

        self.alpha = 0.1
        self.gamma = 1
        self.state_est = {"Lend": 0, "A": 0.1, "B": 0.15, "C": 0.20, "D": 0.25, "E": 0.3, "Rend": 0}
        self.steps = 100
        self.step_list = [el for el in range(1, 101)]

    def random_walk(self):

        self.state_est = {"Lend": 0, "A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5, "E": 0.5, "Rend": 0}
        # np.random.seed(self.steps)
        for i in range(0, self.steps + 1):
            reward = 0
            state = 3
            while 6 > state > 0:
                p = np.random.random()
                if p < 0.5:
                    # left
                    self.state_est[self.path[state]] = self.state_est[self.path[state]] + self.alpha * \
                                                        (reward + self.gamma * self.state_est[self.path[state - 1]] -
                                                         self.state_est[self.path[state]])
                    state -= 1

                else:
                    # right
                    if self.path[state] == "E":
                        reward = 1
                    self.state_est[self.path[state]] = self.state_est[self.path[state]] + self.alpha * \
                                                        (reward + self.gamma * self.state_est[self.path[state + 1]] -
                                                         self.state_est[self.path[state]])
                    state += 1

    def rms(self):
        performance_list = list()
        ideal_dict = [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]
        self.steps = 100
        self.state_est = {"Lend": 0, "A": 0.5, "B": 0.5, "C": 0.5, "D": 0.5, "E": 0.5, "Rend": 0}
        self.step_list = [el for el in range(1, 1 + self.steps)]

        for i in range(1, self.steps + 1):
            reward = 0
            state = 3
            # np.random.seed(int(self.alpha*1000)-1)
            while 6 > state > 0:
                p = np.random.random()
                if p < 0.5:
                    # left
                    self.state_est[self.path[state]] = self.state_est[self.path[state]] + self.alpha * \
                                                        (reward + self.gamma * self.state_est[self.path[state - 1]] -
                                                         self.state_est[self.path[state]])
                    state -= 1

                else:
                    # right
                    if self.path[state] == "E":
                        reward = 1
                    self.state_est[self.path[state]] = self.state_est[self.path[state]] + self.alpha * (
                            reward + self.gamma * self.state_est[self.path[state + 1]] -
                            self.state_est[self.path[state]])
                    state += 1

            Y = list(self.state_est.values())[1:6:]
            # find individual rms and average it for each state
            sqe = 0
            for j in range(0, 5):
                sqe += math.pow(Y[j] - ideal_dict[j], 2)
            mse = sqe / 5
            rmse = math.sqrt(mse)
            performance_list.append(rmse)
        return performance_list

    def plotting(self):
        plt.ylabel("Estimated Value")
        plt.xlabel("State")
        plt.xlim(0, 6)
        plt.ylim(0, 1)

        plt.figure("Random Walk")

        for i in [0, 1, 10, 100]:
            self.steps = i
            self.random_walk()
            X = list(self.state_est.keys())[1:6:]
            Y = list(self.state_est.values())[1:6:]
            X.insert(0, "0")
            Y.insert(0, None)
            plt.scatter(X, Y)
            plt.plot(X, Y, label='step=' + str(i))
        plt.legend()

        plt.figure("RMS for Random Walk")
        plt.ylabel("Performance RMS")
        plt.xlabel("Walks")
        plt.xlim(0, 100)
        plt.ylim(0, 0.3)

        # p_list = []
        # for j in [0.01, 0.03, 0.05, 0.1, 0.15]:
        #     self.alpha = j
        #     for i in range(100):
        #         p_list.append(self.rms())
        #     mean = np.mean(p_list, axis=0)
        #     plt.plot(mean, label='alpha=' + str(j))

        for i in [0.01, 0.03, 0.05, 0.1, 0.15]:
            self.alpha = i
            performance_list = self.rms()
            plt.plot(performance_list, label='alpha=' + str(i))
        plt.legend()

        plt.show()


# class Map:
#     def __init__(self, start):
#         self.start = start
#         self.left = None
#         self.right = None
#
#     def insert(self, left, right):
#         self.left = Map(left)
#         self.right = Map(right)
#         return self.left, self.right
#
#     def right_set(self, right):
#         self.right = Map(right)
#         self.right.left = Map(self.start)
#         return self.get_right()
#         # self.right.left_set(self.start)
#
#     def get_right(self):
#         return self.right
#
#     def get_left(self):
#         return self.left
#
#     def left_set(self, left):
#         self.left = Map(left)



STEP_COUNT = 1000
BANDIT_SETS = 2000


class Bandit:
    def __init__(self):
        self.step = list()
        self.percent_list = list()
        self.epsilons = 0.1
        self.alpha = 0.1
        self.c = [2, 1, 0]
        self.offsets = [0, 1, 5]
        self.q_star = np.random.normal(0, 1, 10)
        self.optimal = np.argmax(self.q_star)
        self.epsilons = [0, 0.01, 0.1]
        self.greedy_results = []
        self.optimistic_results = []
        self.ucb_results = []
        self.color = ["red", "blue", "yellow"]

    def optimistic_initial_val(self):
        action_selection = np.zeros((len(self.epsilons), BANDIT_SETS, STEP_COUNT))

        for i, epsilon in enumerate(self.epsilons):
            for j in range(BANDIT_SETS):
                q_est = np.zeros(10) + self.offsets[i]  # Q_1(a)

                for k in range(STEP_COUNT):
                    g_choose = [ind for ind, q in enumerate(q_est) if q == np.amax(q_est)]
                    non_g_choose = [ind for ind, q in enumerate(q_est) if q != np.amax(q_est)]
                    action = None
                    if np.random.random() < 1 - epsilon or len(non_g_choose) == 0:
                        action = np.random.choice(g_choose)
                    else:
                        action = np.random.choice(non_g_choose)

                    reward = np.random.normal(self.q_star[action], 1)

                    q_est[action] = q_est[action] + self.alpha * (reward - q_est[action])

                    action_selection[i][j][k] = action

        percent_optimal_action = np.mean((action_selection == self.optimal), axis=1)
        self.optimistic_results = percent_optimal_action
        plt.figure("Optimistic Greedy")
        for i, epsilon in enumerate(self.epsilons):
            plt.plot(percent_optimal_action[i], label='epsilon = ' + str(epsilon))
        plt.xlabel('Steps')
        plt.ylabel('% Optimal action')
        plt.legend()

    def greedy_eta_final(self):
        rewards = np.zeros((len(self.epsilons), BANDIT_SETS, STEP_COUNT))
        action_selection = np.zeros((len(self.epsilons), BANDIT_SETS, STEP_COUNT))

        for i, epsilon in enumerate(self.epsilons):
            for j in range(BANDIT_SETS):
                num_action_selected = np.zeros(10)
                reward_per_action = [list() for _ in range(10)]
                q_est = np.zeros(10)  # Q_1(a)

                for k in range(STEP_COUNT):
                    g_choose = [ind for ind, q in enumerate(q_est) if q == np.amax(q_est)]
                    non_g_choose = [ind for ind, q in enumerate(q_est) if q != np.amax(q_est)]

                    action = None
                    if np.random.random() < 1 - epsilon or len(non_g_choose) == 0:
                        action = np.random.choice(g_choose)
                    else:
                        action = np.random.choice(non_g_choose)

                    reward = np.random.normal(self.q_star[action], 1)

                    reward_per_action[action].append(reward)
                    num_action_selected[action] += 1
                    q_est[action] = q_est[action] + self.alpha * (reward - q_est[action])

                    rewards[i][j][k] = reward
                    action_selection[i][j][k] = action

        average_reward = np.mean(rewards, axis=1)
        percent_optimal_action = np.mean((action_selection == self.optimal), axis=1)
        self.greedy_results = percent_optimal_action
        plt.figure("Greedy epsilom")
        for i, epsilon in enumerate(self.epsilons):
            plt.plot(average_reward[i], label='epsilon = ' + str(epsilon))
        plt.xlabel('Steps')
        plt.ylabel('Average reward')
        plt.legend()

        plt.figure("Greedy epsilon %opt act")
        for i, epsilon in enumerate(self.epsilons):
            plt.plot(percent_optimal_action[i], label='epsilon = ' + str(epsilon))
        plt.xlabel('Steps')
        plt.ylabel('% Optimal action')
        plt.legend()

    def upper_conf_bound(self):
        self.offsets = [0, 0, 0]

        rewards = np.zeros((len(self.epsilons), BANDIT_SETS, STEP_COUNT))
        for i, epsilon in enumerate(self.epsilons):
            for j in range(BANDIT_SETS):
                num_action_selected = np.zeros(10)
                q_est = np.zeros(10) + self.offsets[i]  # Q_1(a)

                for k in range(STEP_COUNT):
                    ucb = self.c[i] * np.sqrt(np.log(k + 1) / (num_action_selected + 1e-07))
                    q_fin = q_est + ucb
                    g_choose = [ind for ind, q in enumerate(q_fin) if q == np.amax(q_fin)]
                    non_g_choose = [ind for ind, q in enumerate(q_fin) if q != np.amax(q_fin)]

                    action = None
                    if np.random.random() < 1 - epsilon or len(non_g_choose) == 0:
                        action = np.random.choice(g_choose)
                    else:
                        action = np.random.choice(non_g_choose)

                    reward = np.random.normal(self.q_star[action], 1)
                    num_action_selected[action] += 1
                    q_est[action] = q_est[action] + 1 / num_action_selected[action] * (reward - q_est[action])

                    rewards[i][j][k] = reward

        average_reward = np.mean(rewards, axis=1)
        self.ucb_results = average_reward
        plt.figure("UCB")
        for i, epsilon in enumerate(self.epsilons):
            plt.plot(average_reward[i], label='UCB c =' + str(self.c[i]) + ', epsilon = ' + str(epsilon))
        plt.xlabel('Steps')
        plt.ylabel('Average reward')
        plt.legend()

    def plotting(self):
        self.greedy_eta_final()
        self.optimistic_initial_val()
        self.upper_conf_bound()

        plt.figure('Optimistic vs Realistic')
        plt.ylabel("%Optimal Actions")
        plt.xlabel("Steps")
        plt.xlim(0, STEP_COUNT)
        plt.ylim(0, 1)
        for i, ep in enumerate(self.epsilons):
            plt.plot(self.greedy_results[i], label="Realistic e=" + str(ep))
        for i, ep in enumerate(self.epsilons):
            plt.plot(self.optimistic_results[i], label="Optimistic e=" + str(ep))

        plt.legend()

        plt.figure('UCB vs Epsilon Greedy')
        plt.xlim(0, STEP_COUNT)
        plt.ylim(0, 1.5)
        for i, ep in enumerate(self.epsilons):
            plt.plot(self.greedy_results[i], label="Realistic e=" + str(ep))
        for i, ep in enumerate(self.epsilons):
            plt.plot(self.ucb_results[i], label="Upper Confidence Bound e" + str(ep) + " c=" + str(self.c[i]))
        plt.legend()


# Run this program from a terminal and see the output
# Please make sure that the program you submit can be run from a terminal
def main():
    Bandit().plotting()
    MRP().plotting()


if __name__ == '__main__':
    main()
