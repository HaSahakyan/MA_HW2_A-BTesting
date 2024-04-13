############################### LOGGER
from abc import ABC, abstractmethod
from logs import *  # Assuming logs module is imported here

import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt

logging.basicConfig  # This line seems incomplete and unnecessary, probably a typo

# Initialize logger
logger = logging.getLogger("MAB Application")

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())  # Assuming CustomFormatter is defined in logs module
logger.addHandler(ch)

class Bandit(ABC):
    """
    Abstract class representing a multi-armed bandit problem.
    """

    @abstractmethod
    def __init__(self, p):
        """
        Initializes the bandit with the given probabilities.

        Parameters:
        p (list): List of probabilities for each arm.
        """
        self.p = p

    @abstractmethod
    def __repr__(self):
        """
        Returns a string representation of the bandit.
        """
        return f"{self.__class__.__name__}({self.p})"

    @abstractmethod
    def pull(self):
        """
        Pulls an arm of the bandit and returns the chosen arm.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Updates the bandit after pulling an arm.
        """
        pass

    @abstractmethod
    def experiment(self):
        """
        Runs an experiment with the bandit.
        """
        pass

    @abstractmethod
    def report(self):
        """
        Reports the performance of the bandit.
        """
        pass

class Visualization():
    """
    Class for visualizing multi-armed bandit algorithms.
    """

    def __init__(self, epsilon_greedy, thompson_sampling, trials):
        self.epsilon_greedy = epsilon_greedy
        self.thompson_sampling = thompson_sampling
        self.trials = trials

    def plot1(self):
        """
        Visualizes the performance of each bandit.
        """
        plt.figure(figsize=(10, 6))
        for bandit in range(len(self.epsilon_greedy.p)):
            epsilon_rewards = self.epsilon_greedy.experiment(self.trials)
            thompson_rewards = self.thompson_sampling.experiment(self.trials)
            plt.plot(epsilon_rewards, label=f'Epsilon Greedy Bandit {bandit}')
            plt.plot(thompson_rewards, label=f'Thompson Sampling Bandit {bandit}')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Rewards')
        plt.title('Performance of Each Bandit')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot2(self):
        """
        Compares cumulative rewards and regrets of E-greedy and Thompson Sampling.
        """
        epsilon_rewards = self.epsilon_greedy.experiment(self.trials)
        thompson_rewards = self.thompson_sampling.experiment(self.trials)

        epsilon_regrets = [max(self.epsilon_greedy.p) * t - reward for t, reward in enumerate(epsilon_rewards)]
        thompson_regrets = [max(self.thompson_sampling.p) * t - reward for t, reward in enumerate(thompson_rewards)]

        plt.figure(figsize=(10, 6))
        plt.plot(epsilon_rewards, label='Epsilon Greedy')
        plt.plot(thompson_rewards, label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Rewards')
        plt.title('Comparison of Cumulative Rewards')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(epsilon_regrets, label='Epsilon Greedy')
        plt.plot(thompson_regrets, label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Regrets')
        plt.title('Comparison of Cumulative Regrets')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def save_to_csv(data, filename="rewards.csv"):
        """
        Saves experiment data to a CSV file.

        Parameters:
        data (list): List of experiment data.
        filename (str): Name of the CSV file.
        """
        df = pd.DataFrame(data, columns=["Bandit", "Reward", "Algorithm"])
        df.to_csv(filename, index=False)

class EpsilonGreedy(Bandit):
    """
    Epsilon Greedy multi-armed bandit algorithm.
    """

    def __init__(self, p, e):
        """
        Initializes the Epsilon Greedy bandit with probabilities and epsilon value.

        Parameters:
        p (list): List of probabilities for each arm.
        e (float): Epsilon value.
        """
        self.p = p
        self.e = e
        self.n = len(p)
        self.n_bandit_pull = np.zeros(self.n)
        self.reward_per_bandit = np.zeros(self.n)
        self.t = 0

    def pull(self):
        """
        Chooses an arm based on the epsilon-greedy policy.
        """
        self.e = 1 / (1 + self.t)
        if np.random.random() < self.e:
            chosen_bandit = np.random.choice(range(self.n))
        else:
            chosen_bandit = np.argmax(self.reward_per_bandit / (self.n_bandit_pull + 1e-5))
        return chosen_bandit

    def update(self, bandit):
        """
        Updates the bandit after pulling an arm.

        Parameters:
        bandit (int): The chosen arm.
        """
        self.t += 1
        self.n_bandit_pull[bandit] += 1
        reward = self.p[bandit]
        self.reward_per_bandit[bandit] += reward
        return reward

    def experiment(self, trials):
        """
        Runs an experiment with the Epsilon Greedy bandit.

        Parameters:
        trials (int): Number of trials.

        Returns:
        list: Cumulative rewards for each trial.
        """
        cumulative_rewards = []
        reward_sum = 0
        for _ in range(trials):
            bandit = self.pull()
            reward = self.update(bandit)
            reward_sum += reward
            cumulative_rewards.append(reward_sum)
        return cumulative_rewards

    def report(self):
        """
        Reports the performance of the Epsilon Greedy bandit.
        """
        mean_reward = np.sum(self.reward_per_bandit) / self.t
        optimal_reward = max(self.p) * self.t
        mean_regret = optimal_reward - np.sum(self.reward_per_bandit)
        print(f"Average Reward for EpsilonGreedy: {mean_reward}")
        print(f"Average Regret for EpsilonGreedy: {mean_regret}")

    def __repr__(self):
        """
        Returns a string representation of the Epsilon Greedy bandit.
        """
        return f'EpsilonGreedy algorithm with p = {self.p} and epsilion = {self.e}'

class ThompsonSampling(Bandit):
    """
    Thompson Sampling multi-armed bandit algorithm.
    """

    def __init__(self, p):
        """
        Initializes the Thompson Sampling bandit with probabilities.

        Parameters:
        p (list): List of probabilities for each arm.
        """
        self.p = p
        self.n = len(p)
        self.alpha = np.ones(self.n)
        self.beta = np.ones(self.n)

    def __repr__(self):
        """
        Returns a string representation of the Thompson Sampling bandit.
        """
        return f'ThompsonSampling algorithm with p = {self.p}, alpha = {self.alpha} and beta = {self.beta}'

    def pull(self):
        """
        Chooses an arm based on Thompson Sampling.
        """
        probabilities = [beta.rvs(a, b) for a, b in zip(self.alpha, self.beta)]
        bandit = np.argmax(probabilities)
        return bandit

    def update(self, x):
        """
        Updates the bandit after pulling an arm.

        Parameters:
        x (int): The chosen arm.
        """
        reward = self.p[x]
        self.alpha[x] += reward
        self.beta[x] += max(self.p) - reward
        return reward

    def experiment(self, trials):
        """
        Runs an experiment with the Thompson Sampling bandit.

        Parameters:
        trials (int): Number of trials.

        Returns:
        list: Cumulative rewards for each trial.
        """
        cumulative_rewards = []
        reward_sum = 0
        for _ in range(trials):
            bandit = self.pull()
            reward = self.update(bandit)
            reward_sum += reward
            cumulative_rewards.append(reward_sum)
        return cumulative_rewards

    def report(self):
        """
        Reports the performance of the Thompson Sampling bandit.
        """
        reward = np.sum(self.alpha) - len(self.alpha)
        mean_reward = reward / (np.sum(self.alpha) + np.sum(self.beta) - 2 * len(self.alpha))
        optimal_reward = max(self.p) * (np.sum(self.alpha) + np.sum(self.beta) - 2 * len(self.alpha))
        mean_regret = optimal_reward - reward
        print(f"Average Reward for ThompsonSampling: {mean_reward}")
        print(f"Average Regret for ThompsonSampling: {mean_regret}")

def comparison(epsilon_greedy, thompson_sampling, trials):
    """
    Compares the performance of Epsilon Greedy and Thompson Sampling algorithms.

    Parameters:
    epsilon_greedy (EpsilonGreedy): Instance of Epsilon Greedy bandit.
    thompson_sampling (ThompsonSampling): Instance of Thompson Sampling bandit.
    trials (int): Number of trials for the comparison.
    """
    # Running experiments for both algorithms
    epsilon_rewards = epsilon_greedy.experiment(trials)
    thompson_rewards = thompson_sampling.experiment(trials)

    data = []
    for i, reward in enumerate(epsilon_rewards):
        data.append([i, reward, "Epsilon Greedy Algorithm"])
    for i, reward in enumerate(thompson_rewards):
        data.append([i, reward, "Thompson Sampling Algorithm"])

    Visualization(epsilon_greedy, thompson_sampling, trials).save_to_csv(data)

    # Calculating cumulative regrets
    epsilon_regrets = [max(epsilon_greedy.p) * t - reward for t, reward in enumerate(epsilon_rewards)]
    thompson_regrets = [max(thompson_sampling.p) * t - reward for t, reward in enumerate(thompson_rewards)]

    # Plotting cumulative rewards
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_rewards, label='Epsilon Greedy')
    plt.plot(thompson_rewards, label='Thompson Sampling')
    plt.xlabel('Trials')
    plt.ylabel('Cumulative Rewards')
    plt.title('Comparison of Cumulative Rewards')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting cumulative regrets
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_regrets, label='Epsilon Greedy')
    plt.plot(thompson_regrets, label='Thompson Sampling')
    plt.xlabel('Trials')
    plt.ylabel('Cumulative Regrets')
    plt.title('Comparison of Cumulative Regrets')
    plt.legend()
    plt.grid(True)
    plt.show()

    epsilon_greedy.report()
    thompson_sampling.report()

if __name__ == '__main__':
    Bandit_Reward = [1, 2, 3, 4]
    NumberOfTrials = 20000

    epsilon_greedy = EpsilonGreedy(Bandit_Reward, 0.1)
    thompson_sampling = ThompsonSampling(Bandit_Reward)

    comparison(epsilon_greedy, thompson_sampling, NumberOfTrials)

    # Logging messages
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

       
