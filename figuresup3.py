import numpy as np
import sklearn.decomposition
import scipy.ndimage
import scipy.interpolate
import matplotlib.pyplot as plt

loading1colour = "#2171b5ff"
score1colour = "#ef3b2cff"

# All of this is copied and pasted from the covariance supplemental figure.
N = 100000
T = 30
np.random.seed(2)
white = np.random.randn(T, N)
long_white = np.random.randn(T*3, N)
_random_walk = np.cumsum(white, axis=0)
random_walk = _random_walk - _random_walk[0]

_ar1s = [np.random.randn(N)]
coef = .92
for i in range(0, T*3):
    _ar1s.append(_ar1s[-1]*coef + np.random.randn(N))

ar1s = np.asarray(_ar1s)[T:(T*2)]

timeseries = [
#("No autocorrelation", white, [False,False]),
("Low-pass filtered\nwhite noise", scipy.ndimage.gaussian_filter1d(long_white, axis=0, sigma=4)[T:(2*T)], [False,False]),
("Wrap-around", scipy.ndimage.gaussian_filter1d(white, axis=0, sigma=4, mode="wrap"), ['<', '>']),
("AR process", ar1s, [False,False]),
("MA process", long_white[0:T]+long_white[1:(T+1)]*.75+long_white[2:(T+2)]*.5++long_white[3:(T+3)]*.25, [False,False]),
("Random walk\nstarting at 0", random_walk, ['X', False]),
("Random walk\nmean-subtracted", random_walk-np.mean(random_walk, axis=0, keepdims=True), [False,False]),
("Random walk\nstarting and ending at 0",
        random_walk - np.linspace(0, 1, random_walk.shape[0])[:,None]*(random_walk[-1]),
        ['X', 'X']),
]



# This is all copied from Figure 5
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

tss_examples = [("Simulated smoothness\nfrom Figure 2", tss_fmri),
                ("Resting state fMRI\nfrom Figure 2",   tss_generated),
                ("Shifted timeseries\nfrom Figure 3",   tss_monkey),
                ("Random dot motion\n(stimulus-aligned)\nfrom Figure 4",    tss_shadlen),
                ("Random dot motion\n(choice-aligned)\nfrom Figure 4",    tss_shadlen_sac),
                ]

texts = [e[0] for e in tss_examples] + [e[0]+"\nfrom Figure S2" for e in timeseries]
alltss = [e[1] for e in tss_examples] + [e[1].T for e in timeseries]
allpcs = [sklearn.decomposition.PCA(n_components=3).fit(tss) for tss in alltss]
allpcsrev = [sklearn.decomposition.PCA(n_components=3).fit_transform(tss.T) for tss in alltss]

from cand import Point, Vector, Canvas
c = Canvas(7, 6.2, "in")
c.set_font("Nimbus Sans", size=8, ticksize=7)
plotnames_comps = [f"comp{i}" for i in range(0, len(allpcs))]
c.add_grid(plotnames_comps, 2, Point(.2, 3.4, "in"), Point(6.75, 5.55, "in"), size=Vector(0.8, 0.8, "in"))
plotnames_scores = [f"scores{i}" for i in range(0, len(allpcs))]
c.add_grid(plotnames_scores, 2, Point(.2, 0.1, "in"), Point(6.75, 2.25, "in"), size=Vector(0.8, 0.8, "in"))

for i,(pcs,text) in enumerate(zip(allpcs,texts)):
    ax = c.ax(plotnames_comps[i])
    ax.plot(pcs.components_[0], pcs.components_[1], c=loading1colour, linewidth=2.5)
    ax.set_title(text)
    ax.axis("off")

for i,(pcs,text) in enumerate(zip(allpcsrev,texts)):
    ax = c.ax(plotnames_scores[i])
    ax.scatter(pcs[:,0], pcs[:,1], c=score1colour, s=10)
    ax.set_title(text)
    ax.axis("off")

c.add_text("PC components - PC1 vs PC2", Point(.1, 6.1, "in"), size=10, ha="left", weight="bold")
c.add_text("PC scores (transposed matrix) - PC1 vs PC2", Point(.1, 2.8, "in"), size=10, ha="left", weight="bold")
    
c.save("figuresup3.pdf")
