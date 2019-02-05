from numpy import transpose as tpose
from numpy import loadtxt as ldtxt
from pylab import *
from os import listdir as ld
from os.path import join
import numpy as np

def calculate_avg(mtrx, avg_leng):
    running_avg = []
    routed = []
    dropped = []
    frac_dropped = []

    for i in range(0, len(mtrx[4]), avg_leng):
        temp = np.mean(mtrx[4][i:i + avg_leng])
        running_avg.append(temp)
        temp2 = np.sum(mtrx[3][i:i + avg_leng])
        routed.append(temp2)
        temp3 = np.sum(mtrx[2][i:i + avg_leng])
        dropped.append(temp3)
        fraction_dropped.append( temp2/(temp2 + temp3) )
    return running_avg, frac_dropped


number_of_subplots = 2
avg_len = 60

# rate = input('1.25, 2.5 or 5?: ')
prefix = "/Users/JLibin/Downloads/Paper/constant_interarrival_times/2.5" 
tf1 = ld(prefix)
tf = [join(prefix, tf1[f]) for f in range(len(tf1))]
rltf = len(tf)
transposed = [tpose(ldtxt(tf[f], delimiter=',', usecols=range(5))) for f in range(rltf)]
calc_averages = [calculate_avg(transposed[t], avg_len) for t in range(rltf)]
running_averages = [calc_averages[n][0] for n in range(rltf)] 
fraction_dropped = [calc_averages[n][1] for n in range(rltf)]
pts = [linspace(1, len(running_averages[n]), len(running_averages[n])) for n in range(rltf)]

fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
for n in range(rltf):
    ax1.plot(pts[0], running_averages[n], label='%s' % tf1[n][:-10])
ax1.set_ylabel('Average Reward', fontsize=8)	
ax1.set_xlabel('Iteration', fontsize=8)	
ax1.set_title("Average Reward Learning Curve", fontsize=10)
ax1.legend(loc='lower right', fontsize=8, frameon=False)

ax2 = plt.subplot(2, 1, 2)
for n in range(rltf):
    ax2.plot(pts[0], fraction_dropped[n], label='%s' % tf1[n][:-10])
ax2.legend(loc='upper right', fontsize=8, frameon=False)	
ax2.set_ylabel('Fraction of Requests Dropped', fontsize=8)	
ax2.set_xlabel('Iteration', fontsize=8)	
ax2.set_title("Jobs Dropped Learning Curve", fontsize=10)

fig.subplots_adjust(hspace=0.5)
plt.show()
