import numpy as np
import matplotlib.pyplot as plt


def calculate_avg(matrix, avg_length):
    running_average, routed, dropped = [], [], []

    percent_dropped = []
    for i in range(0, len(matrix[4]), avg_length):
        temp = np.mean(matrix[4][i:i + avg_length])
        running_average.append(temp)
        temp2 = np.sum(matrix[3][i:i + avg_length])
        routed.append(temp2)
        temp3 = np.sum(matrix[2][i:i + avg_length])
        dropped.append(temp3)
        percent_dropped.append(temp2 / (temp2 + temp3))
    return (running_average, percent_dropped)

avg_length = 240
test_file1 = '/Users/JLibin/Desktop/prediction.txt'
a7 = np.loadtxt(test_file1, delimiter=',', usecols=range(5))
b7 = np.transpose(a7)
running_average7, percent_dropped7 = calculate_avg(b7, avg_length)

# test_file2 = 'predictionsTestPar10'
# a8 = np.loadtxt(test_file2, delimiter=',', usecols=range(5))
# b8 = np.transpose(a8)
# running_average8, percent_dropped8 = calculate_avg(b8, avg_length)

x_pts7 = np.linspace(1, len(running_average7), len(running_average7))

x_pts8 = np.linspace(1, len(running_average8), len(running_average8))
plt.plot(x_pts7, running_average7, label='Test 1')
plt.plot(x_pts8, running_average8, label='Test 2')

plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.legend(loc='lower right')
plt.show()

plt.plot(x_pts7, percent_dropped7, label='Test 1')
plt.plot(x_pts8, percent_dropped8, label='Test 2')

plt.xlabel('Iteration')
plt.ylabel('Fraction Dropped ')
plt.legend(loc='upper right')
plt.show()
