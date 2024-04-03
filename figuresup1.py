import matplotlib.pyplot as plt
import numpy as np
import sklearn.decomposition
import seaborn as sns
from pcalib import screeplot, scree_legend
import scipy.stats

TIMESERIES_SET = 1

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica Neue LT Std'],'size': 8})

purple = "#beaed4ff"
score1colour = "#ef3b2cff"
score2colour = "#fb6a4aff"
score3colour = "#fc9272ff"
score4colour = "#fcbba1ff"

loadingcolours = [
    "#2171b5ff",
    "#4292c6ff",
    "#6baed6ff",
    "#9ecae1ff",
]

eigen1colour = "#238b45ff"
eigen2colour = "#41ab5dff"
eigen3colour = "#74c476ff"
eigen4colour = "#a1d99bff"




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

if TIMESERIES_SET == 1:
    timeseries = [
        ("No autocorrelation", white, [False,False]),
        ("Low-pass filtered white noise", scipy.ndimage.gaussian_filter1d(long_white, axis=0, sigma=4)[T:(2*T)], [False,False]),
        ("Wrap-around", scipy.ndimage.gaussian_filter1d(white, axis=0, sigma=4, mode="wrap"), ['<', '>']),
        ("AR process", ar1s, [False,False]),
        ("MA process", long_white[0:T]+long_white[1:(T+1)]*.75+long_white[2:(T+2)]*.5++long_white[3:(T+3)]*.25, [False,False]),
        ("Random walk starting at 0", random_walk, ['X', False]),
        ("Mean-subtracted random walk", random_walk-np.mean(random_walk, axis=0, keepdims=True), [False,False]),
        ("Random walk starting and ending at 0",
            random_walk - np.linspace(0, 1, random_walk.shape[0])[:,None]*(random_walk[-1]),
            ['X', 'X']),
    ]
elif TIMESERIES_SET == 2:
    timeseries = [
        ("Unequal variance",
            scipy.stats.norm(0,1).pdf(np.linspace(-3, 3, T))[:,None] * scipy.ndimage.gaussian_filter1d(white, axis=0, sigma=1),
            [False, False]),
        ("Two distinct sections",
            (scipy.stats.norm(0,1).pdf(np.linspace(-3, 7, T))[:,None] + scipy.stats.norm(0,1).pdf(np.linspace(-7,3,T))[:,None]) * scipy.ndimage.gaussian_filter1d(np.random.randn(T, N), axis=0, sigma=1),
            [False, False]),
        ("Two distinct sections with unequal variance",
            (scipy.stats.norm(0,1).pdf(np.linspace(-3, 7, T))[:,None]*1.5 + scipy.stats.norm(0,1).pdf(np.linspace(-7,3,T))[:,None]) * scipy.ndimage.gaussian_filter1d(np.random.randn(T,N), axis=0, sigma=1),
            [False, False]),
    ]

ROWS = len(timeseries)
COLUMNS = 4
COLS = COLUMNS

f = plt.figure(figsize=(7, ROWS+2))
for i,(name,_tss,(anchor_left,anchor_right)) in enumerate(timeseries):
    tss = _tss.T
    # Example timeseries
    plt.subplot(ROWS,COLS,i*COLS+1)
    plt.plot(tss[0:2].T, c=purple)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    sns.despine(ax=plt.gca())
    plt.axhline(0, c='k', linewidth=.3)
    if i==0:
        plt.title("Example timeseries", pad=20)
    if anchor_left is not False:
        if anchor_left == "<":
            plt.plot([0], [.5*tss[0,0]+.5*tss[0,-1]], c='k', marker=anchor_left, markersize=4)
            plt.plot([0], [.5*tss[1,0]+.5*tss[1,-1]], c='k', marker=anchor_left, markersize=4)
        else:
            plt.plot([0], [0], c='k', marker=anchor_left, markersize=4)
    if anchor_right is not False:
        if anchor_left == "<":
            plt.plot([T-1], [.5*tss[0,0]+.5*tss[0,-1]], c='k', marker=anchor_right, markersize=4)
            plt.plot([T-1], [.5*tss[1,0]+.5*tss[1,-1]], c='k', marker=anchor_right, markersize=4)
        else:
            plt.plot([T-1], [0], c='k', marker=anchor_right, markersize=4)
    if i == len(timeseries)-1:
        plt.xlabel("Time")
    # Add label
    plt.text(-.1, 1.05, name, transform=plt.gca().transAxes, fontsize=10, weight="bold")
    # Covariance matrix
    plt.subplot(ROWS,COLS,i*COLS+2)
    cov = np.cov(tss.T)
    # Plot the square of the correlation matrix so we can see it a little better
    plt.imshow(cov, cmap="PuOr", vmin=-np.max(cov), vmax=np.max(cov))
    if i==0:
        plt.title("Covariance matrix", pad=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    if i == len(timeseries)-1:
        plt.xlabel("Time")
    plt.ylabel("Time")
    # Components
    plt.subplot(ROWS,COLS,i*COLS+3)
    pca = sklearn.decomposition.PCA(n_components=4)
    pca.fit(tss)
    for j in range(0, 4):
        plt.plot(pca.components_[j] * np.sign(pca.components_.T[-3,j]), c=loadingcolours[j])
    if i==0:
        plt.title("Loadings", pad=20)
    plt.gca().set_xticks([])
    if i == len(timeseries)-1:
        plt.xlabel("Time")
    plt.axhline(0, c='k', linewidth=.3)
    plt.gca().set_yticks([])
    sns.despine(ax=plt.gca())
    # Scree
    plt.subplot(ROWS, COLS, i*COLS+4)
    screeplot(tss, None, 10, plt.gca())
    plt.ylabel("Var exp.")
    if i==0:
        plt.title("Scree plot", pad=20)
    plt.gca().set_aspect(1/plt.gca().get_data_ratio())
    plt.gca().set_yticks([])
    if i == len(timeseries)-1:
        plt.xlabel("PC #")
    else:
        plt.gca().set_xticks([])
        plt.xlabel("")

plt.tight_layout()
plt.subplots_adjust(wspace=0.01, hspace=0.4, top=.97-(.03*8/ROWS), right=(1.05 if TIMESERIES_SET==1 else 1.03))
if TIMESERIES_SET == 1:
    plt.savefig("figuresup1.pdf")
elif TIMESERIES_SET == 2:
    plt.savefig("figuresup6.pdf")
