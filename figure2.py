import numpy as np
import matplotlib.pyplot as plt
import colorednoise
from sklearn.decomposition import PCA
from pcalib import *
from cand import Canvas, Point, Vector
import scipy
import os

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

ltext = "Loadings" if not REVERSE else "Scores"
spatialltext = "Loadings" if REVERSE else "Scores"

np.random.seed(0)

# cc = cdatasets.CamCanFiltered()
# tss = cc.get_timeseries()
# rng = np.random.RandomState(0)
# selection = np.asarray([tss[i][j][k:k+60] for i,j,k in zip(list(range(0, tss.shape[0]))*60, list(rng.randint(0, 115, 60*len(tss))), list(rng.randint(0, 200, 60*len(tss))))])
# selection = selection - np.mean(selection, axis=1, keepdims=True)
# np.save("camcan_timeseries.npy", selection)

# with h5py.File("/home/max/Research_data/reprocessed_hcp/timeseries.hdf5", 'r') as f:
#     centroids = np.array(f['parcel_centroid'])
# np.save("fmri_centroids.npy", centroids)


tss_generated = scipy.ndimage.gaussian_filter1d(np.random.randn(100000,300), axis=1, sigma=4)[:,140:200]
cv_tss_generated = scipy.ndimage.gaussian_filter1d(np.random.randn(100000,300), axis=1, sigma=4)[:,140:200]

repeats = 500
N_comps = 2
T = 60
T_short = 50

X = np.arange(0, 60)
positive_control_long = [np.cos(X*k*np.pi/T) for k in range(2, 5)]
positive_control_short = [np.cos(X[0:T_short]*k*np.pi/T_short) for k in range(2, 5)]
negative_control_long = [np.cos(X*k*np.pi/T) for k in range(2, 5)]
negative_control_short = [np.cos(X[0:T_short]*k*np.pi/T) for k in range(2, 5)]
pca = PCA(n_components=4)

if not REVERSE:
    pca.fit(tss_generated)
    generated_components_long = pca.components_
    pca.fit(tss_generated[:,0:50])
    generated_components_short = pca.components_
elif REVERSE:
    generated_components_long = pca.fit_transform(np.asarray(tss_generated).T).T
    generated_components_short = pca.fit_transform(tss_generated[:,0:50].T).T

# START

names_tss = ["tss_generated", "comps_generated", "tss_fmri", "comps_fmri"]
names_scree = ["scree_generated", "scree_fmri"]

c = Canvas(7.0, 6.0, "in")
c.set_font("Nimbus Sans", size=8, ticksize=7)
c.add_grid(names_tss, 2, Point(.5, 2.7, "in"), Point(4.9, 5.5, "in"), size=Vector(2, 1, "in"))
c.add_grid(names_scree, 2, Point(5.6, 2.7, "in"), Point(6.6, 5.5, "in"), size=Vector(1, 1, "in"))
mripoint = Point(4.0, 1.25, "in")
c.add_grid([f"widefield{i}" for i in range(0, 8)], 2, Point(0.5, 0.5, "in"), Point(3.6, 1.8, "in"), size=Vector(.65, .5, "in")) # DO NOT CHANGE SIZE OR THIS WILL SCREW UP THE MRI PLOT


# cc = cdatasets.CamCanFiltered()
# tss = cc.get_timeseries()
# rng = np.random.RandomState(0)
# selection = np.asarray([tss[i][j][k:k+60] for i,j,k in zip(list(range(0, tss.shape[0]))*60, list(rng.randint(0, 115, 60*len(tss))), list(rng.randint(0, 200, 60*len(tss))))])
# selection = selection - np.mean(selection, axis=1, keepdims=True)
# np.save("camcan_timeseries.npy", selection)


tss_generated = scipy.ndimage.gaussian_filter1d(np.random.randn(100000,300), axis=1, sigma=4)[:,140:200]
cv_tss_generated = scipy.ndimage.gaussian_filter1d(np.random.randn(100000,300), axis=1, sigma=4)[:,140:200]

_tss_fmri = np.load("/home/max/Research_data/cortexlab/pca/camcan_timeseries.npy")
_tss_fmri = _tss_fmri[np.random.RandomState(0 if not REVERSE else 1).permutation(_tss_fmri.shape[0])]
tss_fmri = _tss_fmri[::2]
cv_tss_fmri = _tss_fmri[1::2]

def get_component_frequency(comp, debug_plot=False):
    def fit_sin_obj(x, component):
        N = len(component)
        comp_demean = component# - np.mean(component)
        sincomp = x[2]*np.sin(x[0]*np.linspace(0, np.pi*2, N)+x[1])
        #sincomp = sincomp - np.mean(sincomp)
        return np.sum(np.square(component-sincomp))
    res = scipy.optimize.differential_evolution(fit_sin_obj, [(0, 10), (0, np.pi*2), (0, .5)], args=(comp,))
    if debug_plot:
        plt.plot(comp - np.mean(comp))
        plt.plot(res.x[2]*np.sin(res.x[0]*np.linspace(0, np.pi*2, len(comp))+res.x[1]) - np.mean(res.x[2]*np.sin(res.x[0]*np.linspace(0, np.pi*2, len(comp))+res.x[1])))
        plt.show()
    return res.x[0]

freqs = [get_component_frequency(c) for c in pca.components_]

mainplots = [
    (tss_generated, cv_tss_generated, "Smooth artificial timeseries", "generated", 100),
    (tss_fmri, cv_tss_fmri, "Resting state fMRI timeseries", "fmri", 1970),
    ]

for i,(tss,cv_tss,title,axname,TR) in enumerate(mainplots):
    ax = c.ax(f"tss_{axname}")
    ax.cla()
    ax.set_prop_cycle(plt.cycler(color=[purple]))
    ax.set_xlabel("Time (s)")
    inds = [0, 2]
    ax.plot(X*TR/1000, tss[inds].T, linewidth=2.0)
    ax.set_yticks([])
    sns.despine(ax=ax, left=True)
    ax.set_ylabel("Timeseries value")
    if i == 0:
        ax.set_title("Example timeseries", pad=40)
    
    ax = c.ax(f"comps_{axname}")
    ax.cla()
    pca = PCA(n_components=4)
    if not REVERSE:
        pca.fit(tss)
        comps = pca.components_
    elif REVERSE:
        comps = pca.fit_transform(tss.T).T
    ax.set_prop_cycle(plt.cycler(color=loading_colours if not REVERSE else score_colours))
    ax.plot(X*TR/1000, comps.T * np.sqrt(pca.explained_variance_ratio_.T), linewidth=2.0)
    ax.set_yticks([])
    if not REVERSE:
        ax.set_ylabel("PC loading")
        if i == 0:
            ax.set_title("PC loadings", pad=40)
    elif REVERSE:
        ax.set_ylabel("PC scores")
        if i == 0:
            ax.set_title("PC scores", pad=40)
    sns.despine(ax=ax, left=True)
    ax.set_xlabel("Time (s)")
    
    ax = c.ax(f"scree_{axname}")
    if not REVERSE:
        screeplot(tss, cv_tss, 10, ax)
    elif REVERSE:
        screeplot(tss.T, cv_tss.T, 10, ax)
    if i == 0:
        ax.set_title("Scree plot")
    section_label(c, f"tss_{axname}", title)

#section_label(c, f"tss_generated", "Smooth timeseries")

scree_legend(c, "scree_generated")


cmap = plt.cm.colors.LinearSegmentedColormap.from_list('grad', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:000000-50:FFFFFF-100:266D9C
    (0.000, (0.000, 0.000, 0.000)),
    (0.500, (1.000, 1.000, 1.000)),
    (1.000, (0.149, 0.427, 0.612) if REVERSE else (.937, .231, .173))))

#cmap = copy.deepcopy(plt.cm.get_cmap("seismic"))
cmap.set_over((0, 0, 0))

import wbplot
rng = np.random.RandomState(0 if not REVERSE else 1)
centroids = np.load("fmri_centroids.npy")
#dist = scipy.spatial.distance_matrix(centroids[0:180], centroids[0:180])
dist = scipy.spatial.distance_matrix(centroids[0:180], centroids[0:180])
cov = np.exp(-dist/20)
tss = (rng.randn(3000, 180) @ cov).T
tss_cv = (rng.randn(3000, 180) @ cov).T
pca = PCA(n_components=9)
if REVERSE:
    pca.fit(tss.T - np.mean(tss.T, axis=1, keepdims=True))
    #pca.fit(tss.T)
    comps = pca.components_
else:
    comps = pca.fit_transform(tss - np.mean(tss, axis=1, keepdims=True)).T
    #comps = pca.fit_transform(tss).T

sfx = "_scores" if not REVERSE else ""

for comp in range(0, 4):
    if not os.path.isfile(f"mri{comp}{sfx}.png"):
        wbplot.pscalar(f"mri{comp}{sfx}.png", -comps[comp], hemisphere="L", cmap=cmap)
    c.add_image(f"mri{comp}{sfx}.png", mripoint+Vector(1.5, 0, "in")*(comp%2)+Vector(0, -.75, "in")*(comp//2), height=Vector(0, .5, "in"), ha="left", va="bottom", unitname=f"mri{comp}")


def wf_comp_to_img(scores, points):
    points = points - np.min(points, axis=0)+1
    dimx = int(np.max([p[0] for p in points]))+2
    dimy = int(np.max([p[1] for p in points]))+2
    grid = np.zeros((dimx,dimy,9))*np.nan
    for i in range(0, len(points)):
        grid[points[i][0],points[i][1],:] = scores[i][0:9]
    return grid


import scipy.io
dat = scipy.io.loadmat("hemi_mask.mat")
imageL = dat['roi'][0][0][::4,::4]
imageR = dat['roi'][0][1][::4,::4]
m = np.hstack([np.asarray(np.where(imageL)), np.asarray(np.where(imageR))])
cov = np.exp(-scipy.spatial.distance_matrix(m.T, m.T)/5)
tss_widefield = (np.random.RandomState(0 if not REVERSE else 1).randn(10100, m.shape[1]) @ cov).T
pca = PCA(n_components=9)
if REVERSE:
    #scores = pca.fit_transform(tss_widefield.T)
    scores = pca.fit_transform(tss_widefield.T - np.mean(tss_widefield.T, axis=1, keepdims=True))
    scores_img = wf_comp_to_img(pca.components_.T, m.T)
elif not REVERSE:
    #scores = pca.fit_transform(tss_widefield)
    scores = pca.fit_transform(tss_widefield - np.mean(tss_widefield, axis=1, keepdims=True))
    scores_img = wf_comp_to_img(scores, m.T)

border = np.logical_xor(~np.isnan(scores_img[:,:,0]), scipy.ndimage.binary_dilation(~np.isnan(scores_img[:,:,0])))

for i in range(0, 8):
    ax = c.ax(f"widefield{i}")
    ax.cla()
    ax.imshow(np.where(border, border*1000, scores_img[:,:,i]), cmap=cmap, vmax=np.nanmax(scores_img[:,:,i]), vmin=np.nanmin(scores_img[:,:,i]), interpolation="none")
    #ax.imshow(np.isnan(scores_img[:,:,i]), interpolation="none")
    ax.axis("off")
    c.add_text(f"PC {i+1}", Point(.5, 1.2, f"axis_widefield{i}"), size=8)

for i in range(0, 4):
    c.add_text(f"PC {i+1}", Point(.5, 1.2, (f"mri{i}", f"axis_widefield{4*(i//2)}")), size=8)

c.add_figure_labels([("a", "tss_generated"),
                     ("b", "comps_generated"),
                     ("c", "scree_generated"),
                     ("d", "widefield0"),
                     ("e", "widefield0", Vector(3.7, 0, "in"))], size=10)

section_label(c, f"widefield0", f"PC {spatialltext.lower()} for spatially smooth data")

if not REVERSE:
    c.save("figure2.pdf")
else:
    c.save("figuresup2.pdf")
