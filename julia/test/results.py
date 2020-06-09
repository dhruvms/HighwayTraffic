#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.abspath('..'))

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import random

import seaborn as sns
plt.style.use('seaborn')

# plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'font.family': 'serif'})
# plt.rcParams.update({'font.weight': 'semibold'})

FIG, AX = plt.subplots()
MAX_TICKS = 200

def read_file(filename):
    with open(filename, 'r') as f:
        line = f.readline()[:-1]
        count = 0

        data = {}

        while line:
            if count == 0:
                data["merge_tick"] = int(line) * 0.2

            if count == 1:
                data["ep_length"] = int(line)
                if data["ep_length"] <= 0:
                    return data

            if count == 2:
                data["min_dist"] = float(line)

            if count == 3:
                data["avg_offset"] = float(line)

            line = f.readline()[:-1]
            count += 1

        return data

def get_results_matrix(data):
    data = np.array(data)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    data = np.vstack([means, means - stds, means + stds]).T
    return data

def plot_hist(data, xlabel, title, binwidth, colour, filename, FIG, AX):
    AX.spines["top"].set_visible(False)
    AX.spines["right"].set_visible(False)
    AX.get_xaxis().tick_bottom()
    AX.get_yaxis().tick_left()
    AX.set_xticks(np.arange(min(data), max(data) + binwidth, 3 * binwidth),
                    minor=False)
    AX.tick_params(axis="x", labelsize=8)
    AX.tick_params(axis="y", labelsize=8)
    AX.set_xlabel(xlabel, fontsize=12)
    AX.set_ylabel("Count", fontsize=12)
    FIG.suptitle(title, fontsize=16)
    n, _, _ = AX.hist(data,
                    bins=np.arange(min(data), max(data) + binwidth, binwidth),
                    color=colour, edgecolor='black', linewidth=1.2)
    AX.set_yticks(np.arange(0, int(math.ceil(max(n) / 10.0)) * 10 + 5, 5),
                    minor=False)
    FIG.savefig(filename + '.eps', format='eps', bbox_inches="tight")
    FIG.clf()
    AX = FIG.gca()

    return FIG, AX

DATADIR = '../data/'

exps = next(os.walk(DATADIR))[1] # 2/3 lane folders
for i, exp in enumerate(exps):
    other_drivers = next(os.walk(os.path.join(DATADIR, exp)))[1] # mixed/cooperative/aggressive
    for j, drivers in enumerate(other_drivers):
        RESULTSDIR = os.path.join(DATADIR, exp, drivers) + '/results/'
        EXPS_SUCCS = {}
        if not os.path.exists(RESULTSDIR):
            os.makedirs(RESULTSDIR)
        for folder, run, files in os.walk(os.path.join(DATADIR, exp, drivers)):
            if len(files) > 0 and 'results' not in folder:
                EXPNAME = folder.split('/')[-2]
                EXPDIR = RESULTSDIR + EXPNAME + '/'
                if not os.path.exists(EXPDIR):
                    os.makedirs(EXPDIR)

                merge_ticks = []
                ep_lengths = []
                min_dists = []
                avg_offsets = []

                for file in files:
                    try:
                        parsed_data = read_file(os.path.join(folder, file))
                    except Exception as e:
                        print(os.path.join(folder, file))

                    if parsed_data["ep_length"] <= 5:
                        continue

                    # for histograms
                    merge_ticks.append(parsed_data["merge_tick"])
                    ep_lengths.append(parsed_data["ep_length"])
                    min_dists.append(parsed_data["min_dist"])
                    avg_offsets.append(parsed_data["avg_offset"])

                # Sangjae conditions
                successes = np.all([np.array(ep_lengths) > 0,
                                    np.array(merge_ticks) > 0], axis=0)
                num_successes = sum(successes)
                total = len(ep_lengths)

                merge_ticks = list(np.array(merge_ticks)[successes])
                min_dists = list(np.array(min_dists)[successes])
                avg_offsets = list(np.array(avg_offsets)[successes])

                merge_time_mean = np.mean(merge_ticks)
                merge_time_stddev = np.std(merge_ticks)

                min_dist_mean = np.mean(min_dists)
                min_dist_stddev = np.std(min_dists)

                avg_offset_mean = np.mean(avg_offsets)
                avg_offset_stddev = np.std(avg_offsets)

                summary_file = EXPDIR + "summary.dat"
                with open(summary_file, 'w') as f:
                    success_rate = (num_successes/total) * 100
                    EXPS_SUCCS[EXPNAME] = (success_rate, num_successes, total)
                    f.write("%f\n" % success_rate)
                    f.write("%f +- %f\n" % (merge_time_mean, merge_time_stddev))
                    f.write("%f +- %f\n" % (min_dist_mean, min_dist_stddev))
                    f.write("%f +- %f\n" % (avg_offset_mean, avg_offset_stddev))

                metrics_file = EXPDIR + "metrics.dat"
                with open(metrics_file, 'w') as f:
                    f.write("%d\n" % num_successes)
                    f.write("%d\n" % total)

                    for tick in merge_ticks:
                        f.write("%f," % tick)
                    f.write('\n')

                    for dist in min_dists:
                        f.write("%f," % dist)
                    f.write('\n')

                    for offset in avg_offsets:
                        f.write("%f," % offset)
                    f.write('\n')

                # data, xlabel, title, binwidth, colour, filename
                FIG, AX = plot_hist(
                            merge_ticks,
                            "Time to change lane (s)",
                            "Histogram of time taken for lane change",
                            1.0,
                            "#7fc97f",
                            EXPDIR + "time_to_merge",
                            FIG, AX)
                FIG, AX = plot_hist(
                            min_dists,
                            "Minimum distance to other vehicles (m)",
                            "Histogram of minimum distances",
                            0.1,
                            "#beaed4",
                            EXPDIR + "min_dists",
                            FIG, AX)
                FIG, AX = plot_hist(
                            avg_offsets,
                            "Average offset from desired lane (m)",
                            "Histogram of offset from desired lane",
                            0.1,
                            "#fdc086",
                            EXPDIR + "avg_offsets",
                            FIG, AX)

        import csv
        w = csv.writer(open(os.path.join(RESULTSDIR, "EXPS_SUCCS.csv"), "w"))
        for key, val in EXPS_SUCCS.items():
            w.writerow([key, val[0], val[1], val[2]])
