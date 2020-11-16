#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import seaborn as sns
plt.style.use('seaborn')

# plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.serif': 'CMU Serif Extra'})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.weight': 'semibold'})

no_stop_and_go = np.genfromtxt('cars-and-gaps_no.csv', delimiter=',')
stop_and_go = np.genfromtxt('cars-and-gaps_yes.csv', delimiter=',')

cars = np.array([1, 2, 3, 4, 5, 6]) * 15
gaps = (np.array([1.125, 1.5, 2.0, 2.5, 3.0]) * 4.0) - 4.0
newshape = (cars.size, gaps.size)

FIG, AX = plt.subplots(1, 3, figsize=(17, 5))
labels = [('MPC(3.0, 0.5, 2)', 2), ('MPC(3.0, 0.25, 2)', 3), ('Ours', 4)]
for i, (name, col) in enumerate(labels):
	im = np.reshape(stop_and_go[:, col], newshape)
	res = AX[i].imshow(im, cmap=cm.rainbow, vmin=11, vmax=100)
	if i == len(labels) - 1:
		cbar = FIG.colorbar(res, ax=AX[i])
		cbar.set_label('Success Rate (%)', rotation=270, weight='semibold', fontsize=24, labelpad=15)
		cbar.ax.tick_params(labelsize=18)
	AX[i].set_xticks(range(gaps.size))
	AX[i].set_yticks(range(cars.size))

	AX[i].set_xticklabels(gaps, fontsize=18)
	AX[i].set_yticklabels(cars, fontsize=18)

	AX[i].set_xlabel('Min. gap between cars (m)', weight='semibold', fontsize=24)
	AX[i].set_ylabel('Number of cars on road', weight='semibold', fontsize=24)

	AX[i].set_xticks(np.arange(-0.5, gaps.size, 1), minor=True);
	AX[i].set_yticks(np.arange(-0.5, cars.size, 1), minor=True);

	AX[i].grid(which='minor', color='k', linestyle='-', linewidth=1.5)
	AX[i].grid()

	AX[i].set_title(name, weight='semibold', fontsize=26)

FIG.suptitle('Success rates with stop-and-go behaviours (E2)', weight='semibold', fontsize=28, x=0.39, y=1.05)
plt.subplots_adjust(left=0.0, bottom=0.09, right=0.80, top=0.87, wspace=0.0, hspace=0.0)
# plt.show()
plt.savefig('stop-and-go' + '.eps', format='eps', bbox_inches="tight")
plt.clf()
plt.cla()
plt.close()

FIG, AX = plt.subplots(1, 3, figsize=(17, 5))
labels = [('MPC(3.0, 0.5, 2)', 2), ('MPC(3.0, 0.25, 2)', 3), ('Ours', 4)]
for i, (name, col) in enumerate(labels):
	im = np.reshape(no_stop_and_go[:, col], newshape)
	res = AX[i].imshow(im, cmap=cm.rainbow, vmin=11, vmax=100)
	if i == len(labels) - 1:
		cbar = FIG.colorbar(res, ax=AX[i])
		cbar.set_label('Success Rate (%)', rotation=270, weight='semibold', fontsize=24, labelpad=15)
		cbar.ax.tick_params(labelsize=18)
	AX[i].set_xticks(range(gaps.size))
	AX[i].set_yticks(range(cars.size))

	AX[i].set_xticklabels(gaps, fontsize=18)
	AX[i].set_yticklabels(cars, fontsize=18)

	AX[i].set_xlabel('Min. gap between cars (m)', weight='semibold', fontsize=24)
	AX[i].set_ylabel('Number of cars on road', weight='semibold', fontsize=24)

	AX[i].set_xticks(np.arange(-0.5, gaps.size, 1), minor=True);
	AX[i].set_yticks(np.arange(-0.5, cars.size, 1), minor=True);

	AX[i].grid(which='minor', color='k', linestyle='-', linewidth=1.5)
	AX[i].grid()

	AX[i].set_title(name, weight='semibold', fontsize=26)

FIG.suptitle('Success rates without stop-and-go behaviours (E1)', weight='semibold', fontsize=28, x=0.39, y=1.05)
plt.subplots_adjust(left=0.0, bottom=0.09, right=0.80, top=0.87, wspace=0.0, hspace=0.0)
# plt.show()
plt.savefig('no-stop-and-go' + '.eps', format='eps', bbox_inches="tight")
plt.clf()
plt.cla()
plt.close()

