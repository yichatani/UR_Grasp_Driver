'''
    AP vs. dataset scale histogram
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
matplotlib.rc('font',family='Times New Roman')


# data
N = 5 # the number of columns
width = 0.7 # column width
parallel_mean = (66, 57, 43, 27, 15)
parallel_std = (3, 3, 3, 3, 3)
suction_mean = (30, 26, 22, 15, 1)
suction_std = (1, 1, 1, 1, 1)
font_size = 24


# create a figure window
fig = plt.figure(figsize=(8, 8), dpi=120)


# parallel jaw AP histogram
ax = fig.add_subplot(111)
p1 = ax.bar(np.arange(N), parallel_mean, width, label="Parallel jaw", color='yellowgreen', \
                                                 edgecolor=None, yerr=None)
plt.xlabel('Dataset Scale', fontsize=font_size)
plt.ylabel('Parallel-jaw AP', fontsize=font_size)


# suction cup AP scatter-line
ax.scatter(np.arange(N), suction_mean, label="Suction cup", color='purple')
p2, = ax.plot(np.arange(N), suction_mean, label="Suction cup", color='purple')
plt.xlabel('scale ratio', fontsize=font_size)
plt.ylabel('AP', fontsize=font_size)


ax.tick_params(axis='x', labelsize=font_size)
ax.tick_params(axis='y', labelsize=font_size)
plt.xticks(np.arange(N), (1.0, 0.8, 0.6, 0.4, 0.2))
plt.yticks(np.array((0, 15, 30, 45, 60)))
plt.legend(handles=[p1, p2], loc="upper right", fontsize=font_size)
plt.title('Test similar dataset', fontsize=font_size)
plt.show()