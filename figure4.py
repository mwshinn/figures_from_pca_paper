import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pcalib

purple = "#beaed4ff"
score1colour = "#ef3b2cff"
score2colour = "#fb6a4aff"
score3colour = "#fc9272ff"
score4colour = "#fcbba1ff"
score_colours = [score1colour, score2colour, score3colour, score4colour]

loading1colour = "#2171b5ff"
loading2colour = "#4292c6ff"
loading3colour = "#6baed6ff"
loading4colour = "#9ecae1ff"
loading_colours = [loading1colour, loading2colour, loading3colour, loading4colour]

eigen1colour = "#238b45ff"
eigen2colour = "#41ab5dff"
eigen3colour = "#74c476ff"
eigen4colour = "#a1d99bff"


dat = scipy.io.loadmat("/home/max/Downloads/roitman/RoitmanDataCode/T1RT.mat", squeeze_me=True, struct_as_record=False)

coherence = dat['x'][:,2]
dot_direction = dat['x'][:,3] - 1
choice_direction = dat['x'][:,4] - 1
was_monkey_correct = dat['x'][:,5]
time_dots_on = dat['x'][:,6]
time_of_saccade = dat['x'][:,7]
rt = dat['x'][:,8]
unit_id = dat['x'][:,0].astype(int)
units = list(sorted(set(unit_id)))

spikes = dat['s']

def psth(trials=None, binsize=100, align="stimulus", pre=None, post=None, by_cell=True):
    if align == "stimulus":
        base_time = time_dots_on
        bins = np.arange(-(pre if pre is not None else 100), (post if post is not None else 1000)+.0001, binsize)
    else:
        base_time = time_of_saccade
        bins = np.arange(-(pre if pre is not None else 1000), (post if post is not None else 500)+.0001, binsize)
    if trials is None:
        trials = np.ones(6149).astype(bool)
    hists = np.asarray([np.histogram(spikes[trials][i]-base_time[trials][i], bins=bins)[0] for i in range(0, len(spikes[trials]))])
    if by_cell:
        units = np.asarray(list(sorted(set(unit_id[trials]))))
        print(units.shape, hists.shape, trials.shape)
        cell_hists = np.asarray([np.mean(hists[unit_id[trials]==u], axis=0) for u in units])
        return cell_hists
    return hists

from cand import Canvas, Point, Vector
c = Canvas(7.0, 6.8, "in")
c.set_font("Nimbus Sans", size=8, ticksize=7)
c.add_image("roitman-fig.png", Point(.3, 6.6, "in"), height=Vector(0, 36, "mm"), ha="left", va="top", unitname="image")
c.add_axis("diagram1", Point(4.7, 5.5, "in"), Point(5.7, 6.5, "in"))
c.add_axis("diagram2", Point(5.8, 5.5, "in"), Point(6.8, 6.5, "in"))
c.add_grid(["tss_stim", "comps_stim"], 1, Point(.5, 3.7, "in"), Point(4.9, 4.7, "in"), size=Vector(2, 1, "in"))
c.add_axis("scree_stim", Point(5.6, 3.7, "in"), Point(6.6, 4.7, "in"))

c.add_grid(["tss_sac", "comps_sac"], 1, Point(.5, 1.9, "in"), Point(4.9, 2.9, "in"), size=Vector(2, 1, "in"))
c.add_axis("scree_sac", Point(5.6, 1.9, "in"), Point(6.6, 2.9, "in"))
# c.add_grid(["tss_sac", "comps_sac"], 1, Point(.5, 1.9, "in"), Point(3.9, 2.9, "in"), size=Vector(1.5, 1, "in"))
# c.add_axis("scree_sac", Point(4.6, 1.9, "in"), Point(5.6, 1.9, "in"))
# c.add_axis("corrs_sac", Point(6.1, 1.9, "in"), Point(7.1, 1.9, "in"))

c.add_grid([f"corrs{i}_sac" for i in range(1, 5)], 1, Point(.5, .5, "in"), Point(6, 1.5, "in"), size=Vector(1,1,"in"))

pcalib.section_label(c, f"tss_stim", "Ramping activity during decision-making (stimulus-aligned)")
pcalib.section_label(c, f"tss_sac", "Movement-related activity (choice-aligned)")

# Use this one for shift driven phantom oscillations
trials = (choice_direction==0) & (coherence>100)
trajs = psth(align="saccade", binsize=50, by_cell=True, pre=500, post=300, trials=trials)
trajs_t = np.arange(-500, 300, 50)
#plt.plot(trajs[26]); plt.show()
# plt.plot(np.arange(-600, 300, 50), np.mean(trajs, axis=0)); plt.show()
peaks = (np.argmax(trajs, axis=1) - np.where(trajs_t==0)[0]) * (trajs_t[1]-trajs_t[0])

diagram_trials = (choice_direction==0) & (rt > 800)
diagram2_trials = (choice_direction==1) & (rt > 800)
ax = c.ax("diagram1")
ax.cla()
ax.plot(np.arange(0, 800, 50), np.mean(psth(align="stimulus", binsize=50, by_cell=False, pre=0, post=800, trials=diagram_trials), axis=0), c='k', clip_on=False)
ax.plot(np.arange(0, 800, 50), np.mean(psth(align="stimulus", binsize=50, by_cell=False, pre=0, post=800, trials=diagram2_trials), axis=0), c='grey', clip_on=False)
sns.despine(ax=ax, left=True)
ax.axvline(0, c='k', linestyle='--')
ax.set_ylim(0, 3.5)
ax.set_yticks([])
ax.set_xlabel("Time from\nstimulus onset (ms)")
ax = c.ax("diagram2")
ax.cla()
ax.plot(np.arange(-500, 300, 50), np.mean(psth(align="saccade", binsize=50, by_cell=False, pre=500, post=300, trials=diagram_trials), axis=0), c='k', clip_on=False)
ax.plot(np.arange(-500, 300, 50), np.mean(psth(align="saccade", binsize=50, by_cell=False, pre=500, post=300, trials=diagram2_trials), axis=0), c='grey', clip_on=False)
sns.despine(ax=ax, left=True)
ax.axvline(0, c='k', linestyle='--')
ax.set_ylim(0, 3.5)
ax.set_yticks([])
ax.set_xlabel("Time from\nchoice (ms)")

c.add_text("Mean LIP activity", Point(.5, 1.1, "axis_diagram1") | Point(.5, 1.1, "axis_diagram2"))


import sklearn.decomposition
pca = sklearn.decomposition.PCA(n_components=5)
tss_demean = trajs - np.mean(trajs, axis=1, keepdims=True)
np.save("_tss_demean_shadlen_sac.npy", tss_demean)
scores = pca.fit_transform(tss_demean)

ax = c.ax("tss_sac")
ax.cla()
ax.plot(trajs_t, tss_demean[10], c=purple)
ax.plot(trajs_t, tss_demean[0], c=purple)
sns.despine(left=True, ax=ax)
ax.axvline(0, c='k', linestyle='--')
ax.set_yticks([])
ax.set_title("Example timeseries")
ax.set_xlabel("Time from choice (ms)")

ax = c.ax("comps_sac")
ax.cla()
for i in range(0, 3):
    ax.plot(trajs_t, pca.components_[i], c=loading_colours[i])

tss_demean_mean = np.mean(tss_demean, axis=0)
ax.plot(trajs_t, tss_demean_mean/np.sqrt(np.sum(np.square(tss_demean_mean))), c='gray', alpha=.4, linewidth=3, linestyle="--", clip_on=False)
ax.axhline(0, c='k')
ax.axvline(0, c='k', linestyle='--')
sns.despine(left=True, ax=ax)
ax.set_yticks([])
ax.set_title("PC loadings")
ax.set_xlabel("Time from choice (ms)")

import pcalib
ax = c.ax("scree_sac")
sset = np.isin(range(0, tss_demean.shape[0]), np.random.RandomState(1).choice(tss_demean.shape[0], 36, replace=False))
pcalib.screeplot(tss_demean[sset], tss_demean[~sset], ax=ax)
ax.set_title("Scree plot")


from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import splrep, BSpline

for j in [0, 1, 2, 3]:
    ax = c.ax(f"corrs{j+1}_sac")
    ax.cla()
    ax.scatter(peaks, scores[:,j] * np.sqrt(pca.explained_variance_ratio_[j]), zorder=10-j, c=score_colours[j], s=2)
    if j == 0:
        ax.set_ylabel("PC score")
    else:
        pass#ax.sharey(c.ax("corrs1_sac"))
    ax.set_xlabel("Estimated peak (ms)")
    ax.plot(np.linspace(-500, 150, 100), lowess(scores[:,j] * np.sqrt(pca.explained_variance_ratio_[j]), peaks, xvals=np.linspace(-500, 150, 100), frac=.75), c=score_colours[j])
    ax.set_yticks([])
    c.add_text(f"PC {j+1}", Point(.5, .75, f"axis_corrs{j+1}_sac"), size=8)
    sns.despine(ax=ax, left=True)





# Use this for smoothness
trials = rt>800
trajs = psth(align="stimulus", binsize=80, by_cell=False, post=800, pre=0)[trials]
trial_rts = rt[trials]




import sklearn.decomposition
pca = sklearn.decomposition.PCA(n_components=10)
tss_demean_stim = trajs - np.mean(trajs, axis=1, keepdims=True)
np.save("_tss_demean_shadlen.npy", tss_demean_stim)
scores = pca.fit_transform(tss_demean_stim)

ax = c.ax("tss_stim")
ax.cla()
ax.plot(np.arange(0, 800, 80), tss_demean_stim[300], c=purple)
ax.plot(np.arange(0, 800, 80), tss_demean_stim[0], c=purple)
sns.despine(left=True, ax=ax)
ax.axvline(0, c='k', linestyle='--')
ax.set_yticks([])
ax.set_title("Example timeseries")
ax.set_xlabel("Time from stimulus onset (ms)")

ax = c.ax("comps_stim")
ax.cla()
ax.set_prop_cycle(plt.cycler(color=loading_colours))
ax.plot(np.arange(0, 800, 80), pca.components_[0:4].T)
ax.axhline(0, c='k')
ax.axvline(0, c='k', linestyle='--')
sns.despine(left=True, ax=ax)
ax.set_yticks([])
ax.set_title("PC loadings")
ax.set_xlabel("Time from stimulus onset (ms)")

import pcalib
ax = c.ax("scree_stim")
sset = np.isin(range(0, tss_demean_stim.shape[0]), np.random.RandomState(1).choice(tss_demean_stim.shape[0], round(tss_demean_stim.shape[0]*2/3), replace=False))
pcalib.screeplot(tss_demean_stim[sset], tss_demean_stim[~sset], ax=ax)
ax.set_title("Scree plot")
pcalib.scree_legend(c, "scree_stim")

# ax = c.ax("tss_comps")
# plt.imshow(np.cov(trajs.T)); plt.colorbar(); plt.show()

c.add_figure_labels([("a", "image"), ("b", "diagram1"), ("c", "tss_stim"), ("d", "comps_stim"), ("e", "scree_stim"), ("f", "tss_sac"),
                     ("g", "comps_sac"), ("h", "scree_sac"), ("i", "corrs1_sac", Vector(0, -.25, "in"))], size=10)

c.save("figure4.pdf")
