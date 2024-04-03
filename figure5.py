import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pcalib import *
from cand import Canvas, Point, Vector
import scipy
import os
import pcalib


purple = "#beaed4ff"
darkpurple = "#553e75"
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


np.random.seed(0)

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
psth = scipy.io.loadmat("/home/max/Research_data/cortexlab/pca/PSTH_E.mat", struct_as_record=False, squeeze_me=True)
# Four side conditions: left, right, bimanual0, bimanual1 (first char of condition ID)
# Two directions: foward or reverse (second char of condition ID)
# Two starting points: top or bottom (third char of condition ID)
# Total of 16 conditions
# 626 neurons in Monkey E from M1 and caudal PMd

normfr = lambda x : x/(5+np.max(x, axis=0)-np.min(x, axis=0)) # How it is normalised in the paper
tss = normfr(psth['PSTH_merged'][9].FR)[1500:3500] # Use only middle 4 cycles, following the paper
timepoints = psth['PSTH_merged'][9].times[1500:3500]+1
handpos = np.mean(psth['PSTH_merged'][9].handPos, axis=0)
handpos_example = psth['PSTH_merged'][9].handPos[20]
timepoints_all = psth['PSTH_merged'][9].times+1

def get_component_frequency(comp, debug_plot=False):
    def fit_sin_obj(x, component):
        N = len(component)
        comp_demean = component# - np.mean(component)
        sincomp = x[2]*np.sin(x[0]*np.linspace(0, np.pi*2, N)+x[1])
        #sincomp = sincomp - np.mean(sincomp)
        return np.sum(np.square(component-sincomp))
    res = scipy.optimize.differential_evolution(fit_sin_obj, [(0, np.min([10, (len(comp)-1)//2+.001])), (0, np.pi*2), (0, .5)], args=(comp,))
    if debug_plot:
        plt.plot(comp - np.mean(comp))
        plt.plot(res.x[2]*np.sin(res.x[0]*np.linspace(0, np.pi*2, len(comp))+res.x[1]) - np.mean(res.x[2]*np.sin(res.x[0]*np.linspace(0, np.pi*2, len(comp))+res.x[1])))
        plt.show()
    return res.x[0]


from sklearn.decomposition import PCA

pca = PCA(n_components=10)
scores = pca.fit_transform(tss[:,::2].T)

powerspectrum = lambda x : np.mean(np.square(np.abs(np.fft.rfft(x, axis=0))), axis=1)

# Showing power spectrum

# Covariance matrix


### COMPARE TO OTHER DATASETS

_tss_fmri = np.load("/home/max/Research_data/cortexlab/pca/camcan_timeseries.npy")
tss_fmri = _tss_fmri[np.random.RandomState(0).permutation(_tss_fmri.shape[0])]

tss_generated = scipy.ndimage.gaussian_filter1d(np.random.randn(100000,300), axis=1, sigma=4)[:,140:200]
tss_shadlen = np.load("_tss_demean_shadlen.npy")
tss_shadlen_sac = np.load("_tss_demean_shadlen_sac.npy")

rng = np.random.RandomState(0)
data = np.load("tsdata.npy", allow_pickle=True)
tsinterp = scipy.interpolate.interp1d(data[0], data[1])
shifts_tss_monkey = rng.rand(2000)*.3
shifts_cv_tss_monkey = rng.rand(2000)*.3
tss_monkey = np.asarray([tsinterp(np.linspace(-.39, .29, 501)+shifts_tss_monkey[i]) for i in range(0, 2000)])



# Generate PC-by-frequency plots
pca_fmri = PCA(n_components=10)
pca_fmri.fit(tss_fmri)
pca_generated = PCA(n_components=10)
pca_generated.fit(tss_generated)
pca_shadlen = PCA(n_components=9)
pca_shadlen.fit(tss_shadlen)
pca_shadlen_sac = PCA(n_components=10)
pca_shadlen_sac.fit(tss_shadlen_sac)
pca_monkey = PCA(n_components=9)
pca_monkey.fit(tss_monkey)
freqs = [get_component_frequency(pca.components_[i]) for i in range(0, pca.n_components)]
freqs_fmri = [get_component_frequency(pca_fmri.components_[i]) for i in range(0, pca_fmri.n_components)]
freqs_generated = [get_component_frequency(pca_generated.components_[i]) for i in range(0, pca_generated.n_components)]
freqs_shadlen = [get_component_frequency(pca_shadlen.components_[i]) for i in range(0, pca_shadlen.n_components)]
freqs_shadlen_sac = [get_component_frequency(pca_shadlen_sac.components_[i]) for i in range(0, pca_shadlen_sac.n_components)]
freqs_monkey = [get_component_frequency(pca_monkey.components_[i]) for i in range(0, pca_monkey.n_components)]


from cand import Vector, Point, Canvas
c = Canvas(7.0, 8.0, "in")
c.set_font("Nimbus Sans", size=8, ticksize=7)
c.add_grid([a+b for b in ["spectrum", "cov", "freq"] for a in ["ames", "generated", "fmri", "shadlen", "monkey", "shadlen_sac"]], 3, Point(.5, .5, "in"), Point(6.8, 4, "in"), spacing=Vector(.3, .5, "in"))
c.add_axis("tss", Point(.5, 4.9, "in"), Point(2.5, 5.9, "in"))
c.add_axis("pcs", Point(3.0, 4.9, "in"), Point(5.0, 5.9, "in"))
c.add_axis("scree", Point(5.5, 4.9, "in"), Point(6.5, 5.9, "in"))
c.add_axis("schematic", Point(3.7, 6.6, "in"), Point(6.5, 7.5, "in"))

c.add_box(Point(.2, .2, "in"), Point(0.93, 5, ("axis_amescov", "in")), zorder=-100, color=(.92, .92, .92), fill=True, boxstyle='round')
c.add_box(Point(.2, 4.60, "in"), Point(6.8, 7.8, "in"), zorder=-100, color=(.92, .92, .92), fill=True, boxstyle='round')

c.add_image("ames-task.png", Point(.5, 6.3, "in"), height=Vector(0, 1.3, "in"), ha="left", va="bottom", unitname="image")

ax = c.ax("schematic")
ax.cla()
ax.plot(timepoints_all, 1+np.cos(handpos_example*2*np.pi), c='k')
ax.plot(timepoints_all, 3.5+np.sin(handpos_example*2*np.pi), c='k')
ax.fill_betweenx([-.5, 5], timepoints_all[1500], timepoints_all[3500], color='grey', alpha=.3, zorder=-100)
sns.despine(left=True, ax=ax)
ax.set_yticks([1, 3.5])
ax.set_yticklabels(["Vertical position", "Horizontal position"])
ax.set_title("Arm position, example trial")
ax.set_xlabel("Time from go cue (s)")
ax.set_facecolor((1,1,1,0))
ax.tick_params(axis='y', width=0, length=0)
c.add_text("Region to analyze", Point(timepoints_all[3420]+.1, 4.90, "schematic"), size=7, ha="left")

ax = c.ax("tss")
ax.plot(timepoints, tss[:,63]+.45, c=purple)
ax.plot(timepoints, tss[:,55]+.3, c=purple)
ax.plot(timepoints, tss[:,100], c=purple)
ax.plot(timepoints, np.mean(tss, axis=1)+.45, c=darkpurple, linestyle="-")
sns.despine(left=True, ax=ax)
ax.set_xlabel("Time (s)")
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_facecolor((1,1,1,0))
ax.set_title("Example timeseries")

ax = c.ax("scree")
sset = np.isin(range(0, tss.shape[0]), np.random.RandomState(1).choice(tss.shape[0], tss.shape[0]//2, replace=False))
screeplot(tss[sset], tss[~sset], 10, ax)
ax.set_ylabel("Var exp.")
ax.set_title("Scree plot")
ax.set_aspect(1/ax.get_data_ratio())
ax.set_yticks([])
ax.set_xlabel("PC #")
ax.set_facecolor((1,1,1,0))
pcalib.scree_legend(c, "scree")

ax = c.ax("pcs")
for i in range(0, 4):
    ax.plot(timepoints, pca.components_[i], c=loading_colours[i])

ax.set_xlabel("Time (s)")
ax.set_title("PC loadings")
sns.despine(left=True, ax=ax)
ax.set_yticks([])
ax.set_yticklabels([])
ax.axhline(0, linestyle='--', c='k', linewidth=1)
ax.set_facecolor((1,1,1,0))

for i,(key,_tss,title,_freqs,dur) in enumerate([
        ("ames", tss, "Pedal task\nTrue oscillations", freqs, 2),
        ("generated", tss_generated.T, "Simulated smoothness\nSmoothness-driven\nfrom Figure 2", freqs_generated, 6),
        ("fmri", tss_fmri.T, "Resting state fMRI\nSmoothness-driven\nfrom Figure 2", freqs_fmri, 118.2),
        ("shadlen", tss_shadlen.T, "Random dot motion\nSmoothness-driven\nfrom Figure 4", freqs_shadlen, .7),
        ("shadlen_sac", tss_shadlen_sac.T, "Random dot motion\nShift-driven\nfrom Figure 4", freqs_shadlen_sac, .8),
        ("monkey", tss_monkey.T, "Shifted timeseries\nShift-driven\nfrom Figure 3", freqs_monkey, .7)]):
    c.add_text(title, Point(.5, 1, "axis_"+key+"spectrum")+Vector(0, .2, "in"), ha="center", weight="bold", size=7)
    ax = c.ax(key+"spectrum")
    ax.plot(np.arange(1, np.min([20,_tss.shape[0]//2]))/dur, powerspectrum(_tss)[1:][0:np.min([20,_tss.shape[0]//2])-1], c='k')
    sns.despine(ax=ax)
    ax.set_xlabel("Frequency (hz)")
    if i==0:
        ax.set_ylabel("Power")
    ax.set_yticks([])
    ax.set_facecolor((1,1,1,0))
    ax = c.ax(key+"cov")
    # Cov matrix
    cov = np.cov(_tss)
    ax.imshow(cov, cmap="PuOr", vmin=-np.max(cov), vmax=np.max(cov))
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_ylabel("Time")
    ax.set_xlabel("Time")
    # Frequency by component number
    ax = c.ax(key+"freq")
    ax.cla()
    ax.scatter(np.arange(1, 7), _freqs[0:6], c='k', clip_on=False, s=8)
    sns.despine(ax=ax)
    ax.set_xlabel("PC #")
    ax.set_ylim(0, 9)
    ax.set_xticks([0, 2, 4, 6])
    ax.set_yticks([0, 2, 4, 6, 8])
    if i != 0:
        ax.set_yticklabels([])
    if i == 0:
        ax.set_ylabel("# cycles in timeseries")
    ax.set_facecolor((1,1,1,0))

c.add_figure_labels([("a", "image"), ("b", "schematic"), ("c", "tss"), ("d", "pcs"), ("e", "scree"), ("f", "amesspectrum", Vector(-.1, -.2, "in")), ("g", "amescov"), ("h", "amesfreq")], size=10)

c.save("figure5.pdf")
