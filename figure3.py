import numpy as np
import matplotlib.pyplot as plt
import colorednoise
from sklearn.decomposition import PCA
import scipy.stats
import copy
from cand import Canvas, Point, Vector
from pcalib import section_label, image_title, screeplot, scree_legend
import skimage.transform
import seaborn as sns
import skimage.transform
import skimage.io

REVERSE = False

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

#cmap = plt.cm.seismic
global_cmap = plt.cm.colors.LinearSegmentedColormap.from_list('grad', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:000000-50:FFFFFF-100:266D9C
    (0.000, (0.000, 0.000, 0.000)),
    (0.500, (1.000, 1.000, 1.000)),
    (1.000, (0.149, 0.427, 0.612) if not REVERSE else (.937, .231, .173))))

global_cmap_reverse = plt.cm.colors.LinearSegmentedColormap.from_list('grad', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:000000-50:FFFFFF-100:266D9C
    (0.000, (0.000, 0.000, 0.000)),
    (0.500, (.8, .8, .8)),
    (1.000, (0.149, 0.427, 0.612) if REVERSE else (.937, .231, .173))))



# Comes from:
# trials = diplib.get_cell_conditional_activity_by_trial(monkey="Q", coh=70, ps=400, time_range=(-600, 600), align="sample", timebin=10)
#  np.save("tsdata.npy", [trials[0], scipy.signal.savgol_filter(np.mean(trials[1], axis=0), 5, 1)])
rng = np.random.RandomState(0 if not REVERSE else 1)
data = np.load("tsdata.npy", allow_pickle=True)
tsinterp = scipy.interpolate.interp1d(data[0], data[1])
shifts_tss_monkey = rng.rand(2000)*.3
shifts_cv_tss_monkey = rng.rand(2000)*.3
tss_monkey = np.asarray([tsinterp(np.linspace(-.39, .29, 501)+shifts_tss_monkey[i]) for i in range(0, 2000)])
cv_tss_monkey = np.asarray([tsinterp(np.linspace(-.39, .29, 501)+shifts_cv_tss_monkey[i]) for i in range(0, 2000)])
X_monkey = np.linspace(-.39+.15, .29+.15, 501)


def f(b, s, t):
    assert 0<=b and b<=1
    assert 0<=s and s<=1
    assert 0<=t and t<=1
    X = np.linspace(0, 10, 1000)-2*t
    noise = colorednoise.powerlaw_psd_gaussian(1.8,10000)[5000:6000]
    truefunc = scipy.stats.gamma(3).pdf(X) - (s/3+.4)*scipy.stats.gamma(4+b*3).pdf(X)
    return truefunc + noise*.03, np.argmax(truefunc)

rng = np.random.RandomState(0 if not REVERSE else 1)
tss_messy, true_messy_peaks = zip(*[f(rng.random(), rng.random(), rng.random()) for _ in range(0, 1002)])
cv_tss_messy, true_cv_messy_peaks = zip(*[f(rng.random(), rng.random(), rng.random()) for _ in range(0, 1002)])
tss_messy = np.asarray(tss_messy)
cv_tss_messy = np.asarray(cv_tss_messy)
X_messy = np.linspace(0, 8000, 1000)

import imageio
im = skimage.io.imread("neuron_image2.png")[:,:,0]
xshifts = scipy.ndimage.gaussian_filter1d(rng.randn(10000), 5)*15
yshifts = scipy.ndimage.gaussian_filter1d(rng.randn(10000), 5)*15
xshifts_cv = scipy.ndimage.gaussian_filter1d(rng.randn(10000), 5)*15
yshifts_cv = scipy.ndimage.gaussian_filter1d(rng.randn(10000), 5)*15

ims_shifted = [scipy.ndimage.shift(im, (xshifts[i],yshifts[i]), mode='nearest')[40:-20,40:-20] for i in range(0, len(xshifts))]
ims_flat = [np.asarray(im_shifted).flatten() for im_shifted in ims_shifted]

ims_shifted_cv = [scipy.ndimage.shift(im, (xshifts_cv[i],yshifts_cv[i]), mode='nearest')[40:-20,40:-20] for i in range(0, len(xshifts_cv))]
ims_flat_cv = [np.asarray(im_shifted).flatten() for im_shifted in ims_shifted_cv]

c = Canvas(7, 5.7, "in")
names_tss = ["tss0", "comps0", "tss1", "comps1"]
names_scree = ["scree0", "scree1"]
names_corrs = ["time_shift_corr0", "time_shift_corr1"]
c.set_font("Nimbus Sans", size=8, ticksize=7)
c.add_grid(names_tss, 2, Point(.3, 2.4, "in"), Point(3.6, 5.2, "in"), size=Vector(1.5, 1, "in"))
c.add_grid(names_scree, 2, Point(4.4, 2.4, "in"), Point(5.4, 5.2, "in"), size=Vector(1, 1, "in"))
c.add_grid(names_corrs, 2, Point(5.9, 2.4, "in"), Point(6.9, 5.2, "in"), size=Vector(1, 1, "in"))
base_start = Point(.8, .5, "in")
base_size = Vector(2, .5, "in")
pcs_size = Vector(1.5, 1, "in")
pcs_offset = Vector(2.4, 0, "in")
scores_offset = Vector(3.6, 0, "in")
c.add_axis("image0", base_start+Vector(0, .5, "in"), base_start+base_size+Vector(0, .5, "in"))
c.add_axis("image_shifts", base_start, base_start+base_size)
c.add_axis("imagepcs0", base_start+pcs_offset, base_start+pcs_offset+pcs_size)
other_offset = Vector(4.5, 0, "in")
c.add_axis("imagescores0", base_start+other_offset, base_start+pcs_size+other_offset)


for i,tss,cv_tss,shifts_tss,X in [(0,tss_monkey,cv_tss_monkey,shifts_tss_monkey,X_monkey), (1,tss_messy,cv_tss_messy,np.asarray(true_messy_peaks)/100,X_messy/1000)]:
    # tss = np.asarray([scipy.stats.gamma(2, rng.randn()*2+10).pdf(np.linspace(0, 30, 501)) for t in range(0, 2000)])
    # cv_tss = np.asarray([scipy.stats.gamma(2, rng.randn()*2+10).pdf(np.linspace(0, 30, 501)) for _ in range(0, 2000)])
    ax = c.ax(f"tss{i}")
    ax.cla()
    ax.set_prop_cycle(plt.cycler(color=[purple]))
    ax.set_xlabel("Time (s)")
    if i == 0:
        inds = [39,21,29]
        base = tss[11]
    elif i == 1:
        inds = [6, 8, 4]
        base = np.mean(tss, axis=0)
    ax.plot(X, tss[inds].T)
    ax.plot(X, base, c='gray', alpha=.4, linewidth=3, linestyle="--", clip_on=False)
    ax.set_yticks([])
    sns.despine(ax=ax, left=True)
    ax.set_ylabel("Timeseries value")
    if i == 0:
        ax.set_title("Example timeseries")
    
    ax = c.ax(f"comps{i}")
    ax.cla()
    pca = PCA(n_components=4)
    if not REVERSE:
        scores = pca.fit_transform(tss)
        comps = pca.components_
    elif REVERSE:
        comps = pca.fit_transform(tss.T).T
        scores = pca.components_.T
    ax.set_prop_cycle(plt.cycler(color=loading_colours if not REVERSE else score_colours))
    ax.plot(X, comps.T * np.sqrt(pca.explained_variance_ratio_.T))
    ax.set_yticks([])
    ax.set_ylabel("PC loading" if not REVERSE else "PC score")
    sns.despine(ax=ax, left=True)
    ax.set_xlabel("Time (s)")
    axlim = ax.axis()
    ax.plot(X, (base-np.mean(base))*(.6 if i==0 else .3), c='gray', alpha=.4, linewidth=3, linestyle="--", clip_on=False)
    ax.axis(axlim)
    if i == 0:
        ax.set_title("PC loadings" if not REVERSE else "PC scores")
    
    ax = c.ax(f"scree{i}")
    ax.cla()
    if not REVERSE:
        screeplot(tss, cv_tss, 10, ax, rng)
    else:
        screeplot(tss.T, cv_tss.T, 10, ax, rng)
    if i == 0:
        scree_legend(c, "scree0", -.15)
        ax.set_title("Scree plot")
    
    ax = c.ax(f"time_shift_corr{i}")
    ax.cla()
    o = np.argsort(shifts_tss)
    ax.set_prop_cycle(plt.cycler(color=score_colours if not REVERSE else loading_colours))
    for j in ([0, 1, 2] if i == 0 else [0, 1]):
        ax.scatter(np.asarray(shifts_tss)[o]-.15, scores[o,j] * np.sqrt(pca.explained_variance_ratio_[j]), s=.1, zorder=10-j)
    if i == 0:
        ax.set_xlabel("Time shift (s)")
    else:
        ax.set_xlabel("Estimated peak (s)")
    ax.set_ylabel("PC score" if not REVERSE else "PC loading")
    ax.set_yticks([])
    sns.despine(ax=ax, left=True)


section_label(c, f"tss0", "Random shifts in timing")
section_label(c, f"tss1", "Random shifts in time with non-identical signals")



def make_grid_4(ims, ax=ax, cm="color"):
    grid = np.zeros((*ims_shifted[0].shape,5))*np.nan
    for i in range(0, 5):
        grid[:,:,i] = ims[i].reshape(ims_shifted[0].shape)
    if cm == "color":
        cmap = copy.deepcopy(global_cmap)
    elif cm == "reverse":
        cmap = copy.deepcopy(global_cmap_reverse)
    elif cm == "bw":
        cmap = copy.deepcopy(plt.cm.gray)
    cmap.set_over((1, 1, 1))
    cmap.set_under(cmap(0))
    gridmin = np.nanmin(grid, axis=(0,1))
    gridmax = np.nanmax(grid, axis=(0,1))
    grid  = (grid-gridmin)/(gridmax-gridmin)
    vline = np.zeros_like(grid[:,0,0])+10000
    hline = np.asarray([10000]*(ims_shifted[0].shape[1]*2+1))
    disp = np.vstack([np.hstack([grid[:,:,0], vline[:,None], grid[:,:,1]]),
                      hline,
                      np.hstack([grid[:,:,2], vline[:,None], grid[:,:,3]]),
])
    ax.imshow(disp, cmap=cmap, vmin=-.000001, vmax=1.000001)
    ax.axis("off")

def make_grid_line_4(ims, ax=ax, cm="color"):
    grid = np.zeros((*ims_shifted[0].shape,5))*np.nan
    for i in range(0, 5):
        grid[:,:,i] = ims[i].reshape(ims_shifted[0].shape)
    if cm == "color":
        cmap = copy.deepcopy(global_cmap)
    elif cm == "reverse":
        cmap = copy.deepcopy(global_cmap_reverse)
    elif cm == "bw":
        cmap = copy.deepcopy(plt.cm.gray)
    cmap.set_over((1, 1, 1))
    cmap.set_under(cmap(0))
    gridmin = np.nanmin(grid, axis=(0,1))
    gridmax = np.nanmax(grid, axis=(0,1))
    grid  = (grid-gridmin)/(gridmax-gridmin)
    vline = np.zeros_like(grid[:,0,0])+10000
    hline = np.asarray([10000]*(ims_shifted[0].shape[1]*2+1))
    disp = np.hstack([grid[:,:,0], vline[:,None], grid[:,:,1], vline[:,None], grid[:,:,2], vline[:,None], grid[:,:,3]])
    ax.imshow(disp, cmap=cmap, vmin=-.000001, vmax=1.000001)
    ax.axis("off")

def make_grid_6(ims, ax=ax, cm="color"):
    grid = np.zeros((*ims_shifted[0].shape,7))*np.nan
    for i in range(0, 7):
        grid[:,:,i] = ims[i].reshape(ims_shifted[0].shape)
    if cm == "color":
        cmap = copy.deepcopy(global_cmap)
    elif cm == "reverse":
        cmap = copy.deepcopy(global_cmap_reverse)
    elif cm == "bw":
        cmap = copy.deepcopy(plt.cm.gray)
    cmap.set_over((1, 1, 1))
    cmap.set_under(cmap(0))
    gridmin = np.nanmin(grid, axis=(0,1))
    gridmax = np.nanmax(grid, axis=(0,1))
    grid  = (grid-gridmin)/(gridmax-gridmin)
    vline = np.zeros_like(grid[:,0,0])+10000
    hline = np.asarray([10000]*(ims_shifted[0].shape[1]*3+2))
    disp = np.vstack([np.hstack([grid[:,:,0], vline[:,None], grid[:,:,1], vline[:,None], grid[:,:,2]]),
                      hline,
                      np.hstack([grid[:,:,3], vline[:,None], grid[:,:,4], vline[:,None], grid[:,:,5]]),
])
    ax.imshow(disp, cmap=cmap, vmin=-.000001, vmax=1.000001)
    ax.axis("off")

def make_grid_8(ims, ax=ax, cm="color"):
    grid = np.zeros((*ims_shifted[0].shape,9))*np.nan
    for i in range(0, 9):
        grid[:,:,i] = ims[i].reshape(ims_shifted[0].shape)
    if cm == "color":
        cmap = copy.deepcopy(global_cmap)
    elif cm == "reverse":
        cmap = copy.deepcopy(global_cmap_reverse)
    elif cm == "bw":
        cmap = copy.deepcopy(plt.cm.gray)
    cmap.set_over((1, 1, 1))
    cmap.set_under(cmap(0))
    gridmin = np.nanmin(grid, axis=(0,1))
    gridmax = np.nanmax(grid, axis=(0,1))
    grid  = (grid-gridmin)/(gridmax-gridmin)
    vline = np.zeros_like(grid[:,0,0])+10000
    hline = np.asarray([10000]*(ims_shifted[0].shape[1]*4+3))
    disp = np.vstack([np.hstack([grid[:,:,0], vline[:,None], grid[:,:,1], vline[:,None], grid[:,:,2], vline[:,None], grid[:,:,3]]),
                      hline,
                      np.hstack([grid[:,:,4], vline[:,None], grid[:,:,5], vline[:,None], grid[:,:,6], vline[:,None], grid[:,:,7]]),
])
    ax.imshow(disp, cmap=cmap, vmin=-.000001, vmax=1.000001)
    ax.axis("off")


pca = PCA(n_components=10)
_scores = pca.fit_transform(ims_flat) if REVERSE else pca.fit_transform(np.asarray(ims_flat).T)
comps = pca.components_ if REVERSE else _scores.T
scores = _scores if REVERSE else pca.components_.T

make_grid_6(comps, c.ax(f"imagepcs0"), "reverse")
make_grid_line_4(np.asarray(ims_shifted)[[0,1000,4000,8000,9000]], c.ax(f"image0"), "bw")


ax = c.ax(f"imagescores0")
width = 28
ax.cla()
for row in [0, 1, 2]:
    for col in [0, 1]:
        ax.scatter(xshifts+row*width, yshifts+col*width, c=scores[:,col*3+row], s=.2, cmap=global_cmap, rasterized=True)

ax.invert_yaxis()
sns.despine(ax=ax, left=True, bottom=True)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("Horizontal shift")
ax.set_ylabel("Vertical shift")

ax = c.ax("image_shifts")
ax.cla()
ax.plot(xshifts[0:500]+20, c='k')
ax.plot(yshifts[0:500], c='k')
ax.set_yticks([0, 20])
ax.set_yticklabels(["Horizontal shift", "Vertical shift"])
sns.despine(left=True, ax=ax)
ax.set_xlabel("Time (ms)")

image_title(c, f"image0", "Registered images")
image_title(c, f"imagepcs0", "PC loadings" if REVERSE else "PC scores")
image_title(c, f"imagescores0", "PC scores" if REVERSE else "PC loadings")



section_label(c, f"image0", "Random shifts in registration", offset=Vector(-.3, 0, "in"))

c.add_figure_labels([("a", "tss0"), ("b", "tss1"), ("c", "scree0"), ("d", "scree1"), ("e", "time_shift_corr0"), ("f", "time_shift_corr1"),
                     ("g", "image0"), ("h", "imagepcs0"), ("i", "imagescores0")], size=10)

sfx = "_scores" if REVERSE else ""
if not REVERSE:
    c.save("figure3.pdf")
else:
    c.save("figuresup5.pdf")
