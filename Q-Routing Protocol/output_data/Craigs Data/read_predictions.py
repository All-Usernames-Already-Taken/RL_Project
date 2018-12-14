import numpy as np
import matplotlib.pyplot as plt


def calculate_avg(matrix, avg_len):
    running_Avg = []
    routed = []
    dropped = []

    percent_dropped = []
    for i in range(0, len(matrix[4]), avg_len):
        temp = np.mean(matrix[4][i:i + avg_len])
        running_Avg.append(temp)
        temp2 = np.sum(matrix[3][i:i + avg_len])
        routed.append(temp2)
        temp3 = np.sum(matrix[2][i:i + avg_len])
        dropped.append(temp3)
        percent_dropped.append(temp2 / (temp2 + temp3))
    return (running_Avg, percent_dropped)


avg_len = 240
test_file1 = 'predictionsTestPar9'
a7 = np.loadtxt(test_file1, delimiter=',', usecols=range(5))
b7 = np.transpose(a7)
running_Avg7, percent_dropped7 = calculate_avg(b7, avg_len)

test_file2 = 'predictionsTestPar10'
a8 = np.loadtxt(test_file2, delimiter=',', usecols=range(5))
b8 = np.transpose(a8)
running_Avg8, percent_dropped8 = calculate_avg(b8, avg_len)

x_pts7 = np.linspace(1, len(running_Avg7), len(running_Avg7))

x_pts8 = np.linspace(1, len(running_Avg8), len(running_Avg8))
plt.plot(x_pts7, running_Avg7, label='Test 1')
plt.plot(x_pts8, running_Avg8, label='Test 2')

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
