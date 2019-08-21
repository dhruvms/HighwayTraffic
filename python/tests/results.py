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

from external import get_args
from utils import get_model_name

def get_colors(n):
  ret = []
  r = int(random.random() * 256)
  g = int(random.random() * 256)
  b = int(random.random() * 256)
  step = 256 / n
  for i in range(n):
    r += step
    g += step
    b += step
    r = int(r) % 256
    g = int(g) % 256
    b = int(b) % 256
    ret.append((r/255.0, g/255.0, b/255.0))
  return ret

def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None, zorder=0, axobj=None):
    if axobj is None:
        # plot the shaded range of the confidence intervals
        plt.fill_between(range(mean.shape[0]), ub, lb,
                         color=color_shading, alpha=.2, zorder=zorder)
        # plot the mean on top
        plt.plot(mean, color=color_mean, zorder=zorder)
    else:
        # plot the shaded range of the confidence intervals
        axobj.fill_between(range(mean.shape[0]), ub, lb,
                         color=color_shading, alpha=.2, zorder=zorder)
        # plot the mean on top
        axobj.plot(mean, color=color_mean, zorder=zorder)

class LegendObject(object):
    def __init__(self, facecolor='red', edgecolor='white', dashed=False):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.facecolor,
            # and whose edges are the faded color
            edgecolor=self.edgecolor, lw=3)
        handlebox.add_artist(patch)

        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2*width/5, y0], width/5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)

        return patch

def read_file(filename):
    UAVs = [0, 2, 5]
    colors = {0: 'r', 2: 'b', 5: 'm'}
    States = {0: [], 2: [], 5: []}
    Goals = {0: [], 2: [], 5: []}
    Waypoints = {0: [], 2: [], 5: []}
    Buffers = {0: [], 2: [], 5: []}
    Map = []

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

            if count == 4:
                data["steer_angle"] = [float(x) for x in line.split(',')[1:]]
                if len(data["steer_angle"]) < 201:
                    data["steer_angle"] += [0.0] *\
                                                (201 - len(data["steer_angle"]))

            if count == 5:
                data["steer_rate"] = [float(x) for x in line.split(',')[1:]]
                if len(data["steer_rate"]) < 201:
                    data["steer_rate"] += [0.0] *\
                                                (201 - len(data["steer_rate"]))

            if count == 6:
                data["vel"] = [float(x) for x in line.split(',')[1:]]
                if len(data["vel"]) < 201:
                    data["vel"] += [0.0] * (201 - len(data["vel"]))

            if count == 7:
                data["jerk"] = [float(x) for x in line.split(',')[1:]]
                if len(data["jerk"]) < 201:
                    data["jerk"] += [0.0] * (201 - len(data["jerk"]))

            if count == 8:
                data["acc"] = [float(x) for x in line.split(',')[1:]]
                if len(data["acc"]) < 201:
                    data["acc"] += [0.0] * (201 - len(data["acc"]))

            if count == 9:
                data["offset"] = [float(x) for x in line.split(',')[1:]]
                if len(data["offset"]) < 201:
                    data["offset"] += [0.0] * (201 - len(data["offset"]))

            if count == 10:
                if line == 'NONE':
                    data["victim"] = {}
                    return data

                data["victim"] = {}
                data["victim"]["alon"] = [float(x) for x in line.split(',')[1:]]
                if len(data["victim"]["alon"]) < 201:
                    data["victim"]["alon"] += [0.0] *\
                                            (201 - len(data["victim"]["alon"]))

            if count == 11:
                data["victim"]["vel"] = [float(x) for x in line.split(',')[1:]]
                if len(data["victim"]["vel"]) < 201:
                    data["victim"]["vel"] += [0.0] *\
                                            (201 - len(data["victim"]["vel"]))

            if count == 12:
                data["victim"]["alat"] = [float(x) for x in line.split(',')[1:]]
                if len(data["victim"]["alat"]) < 201:
                    data["victim"]["alat"] += [0.0] *\
                                            (201 - len(data["victim"]["alat"]))

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

args = get_args()
if args.model_name is None:
    args.model_name = [0, 1, 2, 3, 4, 5, 6, 7, 8]

model_name = get_model_name(args)
modeldir = args.save_dir + model_name + '/'

data_dir = modeldir + 'vids/' # actually inside a timestamped folder inside vids/
results_dir = modeldir + 'results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

exps = next(os.walk(data_dir))[1]
for i, exp in enumerate(exps):
    exp_dir = results_dir + exp + '/'
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    filenames = os.listdir(data_dir + exp)
    merge_ticks = []
    ep_lengths = []
    min_dists = []
    avg_offsets = []

    steer_angles = []
    steer_rates = []
    vels = []
    jerks = []
    accs = []
    offsets = []

    victim_alons = []
    victim_vels = []
    victim_alats = []

    for file in filenames:
        parsed_data = read_file(os.path.join(data_dir, exp, file))

        if parsed_data["ep_length"] <= 0:
            continue

        # for histograms
        merge_ticks.append(parsed_data["merge_tick"])
        ep_lengths.append(parsed_data["ep_length"])
        min_dists.append(parsed_data["min_dist"])
        avg_offsets.append(parsed_data["avg_offset"])

        # for plots over time
        steer_angles.append(parsed_data["steer_angle"])
        steer_rates.append(parsed_data["steer_rate"])
        vels.append(parsed_data["vel"])
        jerks.append(parsed_data["jerk"])
        accs.append(parsed_data["acc"])
        offsets.append(parsed_data["offset"])

        # for plots over time, if exists
        if parsed_data["victim"]:
            victim_alons.append(parsed_data["victim"]["alon"])
            victim_vels.append(parsed_data["victim"]["vel"])
            victim_alats.append(parsed_data["victim"]["alat"])

    successes = sum(np.all([np.array(ep_lengths) == 200,
                        np.array(merge_ticks) > 0,
                        np.array(merge_ticks) < 200], axis=0))
    total = len(ep_lengths)

    succ_idx = np.all([np.array(merge_ticks) > 0,
                        np.array(merge_ticks) < 200], axis=0)
    merge_ticks = list(np.array(merge_ticks)[succ_idx])
    min_dists = list(np.array(min_dists)[succ_idx])
    avg_offsets = list(np.array(avg_offsets)[succ_idx])

    merge_time_mean = np.mean(merge_ticks)
    merge_time_stddev = np.std(merge_ticks)

    min_dist_mean = np.mean(min_dists)
    min_dist_stddev = np.std(min_dists)

    avg_offset_mean = np.mean(avg_offsets)
    avg_offset_stddev = np.std(avg_offsets)

    summary_file = exp_dir + "summary.dat"
    with open(summary_file, 'w') as f:
        success_rate = (successes/total) * 100
        f.write("%f\n" % success_rate)
        f.write("%f +- %f\n" % (merge_time_mean, merge_time_stddev))
        f.write("%f +- %f\n" % (min_dist_mean, min_dist_stddev))
        f.write("%f +- %f\n" % (avg_offset_mean, avg_offset_stddev))

    metrics_file = exp_dir + "metrics.dat"
    with open(metrics_file, 'w') as f:
        f.write("%d\n" % successes)
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

    # input sizes: (total episodes x 201)
    # output sizes: (201 x 3)
    steer_angles_mat = get_results_matrix(steer_angles)
    steer_rates_mat = get_results_matrix(steer_rates)
    vels_mat = get_results_matrix(vels)
    jerks_mat = get_results_matrix(jerks)
    accs_mat = get_results_matrix(accs)
    offsets_mat = get_results_matrix(offsets)

    np.save(exp_dir + "steer_angles", steer_angles_mat)
    np.save(exp_dir + "steer_rates", steer_rates_mat)
    np.save(exp_dir + "vels", vels_mat)
    np.save(exp_dir + "jerks", jerks_mat)
    np.save(exp_dir + "accs", steer_angles_mat)
    np.save(exp_dir + "offsets", offsets_mat)

    # data, xlabel, title, binwidth, colour, filename
    FIG, AX = plot_hist(
                merge_ticks,
                "Time to change lane (s)",
                "Histogram of time taken for lane change",
                1.0,
                "#7fc97f",
                exp_dir + "time_to_merge",
                FIG, AX)
    FIG, AX = plot_hist(
                min_dists,
                "Minimum distance to other vehicles (m)",
                "Histogram of minimum distances",
                0.1,
                "#beaed4",
                exp_dir + "min_dists",
                FIG, AX)
    FIG, AX = plot_hist(
                avg_offsets,
                "Average offset from desired lane (m)",
                "Histogram of offset from desired lane",
                0.1,
                "#fdc086",
                exp_dir + "avg_offsets",
                FIG, AX)

# merge ticks: 7fc97f
# min dists: beaed4
# avg offsets: fdc086
