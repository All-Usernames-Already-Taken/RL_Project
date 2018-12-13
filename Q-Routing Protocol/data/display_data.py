import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    filename = '/Users/joshrutta/Desktop/Fall 2018/Reinforcement Learning/RL Project/RL_Group_Project/Q-Routing Protocol/data/results-2018-12-12 21:27.csv'
    data_df = pd.read_csv(filename)
    reward_data = data_df['calculated_reward'].values
    plt.plot(reward_data)
    plt.show()

if __name__ == '__main__':
    main()