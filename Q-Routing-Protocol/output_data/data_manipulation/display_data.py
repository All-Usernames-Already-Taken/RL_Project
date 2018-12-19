import pandas as pd
import matplotlib.pyplot as plt


def main():
    filename = '/Users/joshrutta/Desktop/Fall 2018/Reinforcement Learning/RL Project/RL_Group_Project/Q-Routing-Protocol/output_data/results.csv'
    data_df = pd.read_csv(filename)
    reward_data = data_df['calculated_reward'].values
    plt.plot(reward_data)
    plt.show()

if __name__ == '__main__':
    main()