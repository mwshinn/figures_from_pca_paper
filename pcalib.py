import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from cand import Canvas, Point, Vector
#from mpl_toolkits import mplot3d

eigen1colour = "#238b45ff"
eigen2colour = "#41ab5dff"
eigen3colour = "#74c476ff"
eigen4colour = "#a1d99bff"

def screeplot(tss, cv_tss=None, n=10, ax=plt.gca(), rng=np.random):
    pca = PCA(n_components=n)
    pca.fit(tss)
    if cv_tss is not None:
        cv_tss = cv_tss - np.mean(cv_tss, axis=0, keepdims=True)
        #scores = (pca.components_ @ tss.T).T
        cv_scores = (pca.components_ @ cv_tss.T).T
        #var_exp = [1-np.var(tss-np.mean(tss, axis=1)-scores[:,i:i+1] @ pca.components_[i:i+1])/np.var(tss) for i in range(0, n)]
        cv_var_exp = [1-np.var(cv_tss-(cv_scores[:,i:i+1] @ pca.components_[i:i+1]))/np.var(cv_tss) for i in range(0, n)]
    gauss_tss = rng.randn(*tss.shape)
    gauss_pca = PCA(n_components=n)
    gauss_pca.fit(gauss_tss)
    x = list(range(1, n+1))
    ax.plot(x, gauss_pca.explained_variance_ratio_, label="White noise", c='gray', linewidth=2, alpha=.7, clip_on=False)
    ax.plot(x, pca.explained_variance_ratio_, label="Fraction var explained", c=eigen1colour, linewidth=1.5)
    if cv_tss is not None:
        ax.plot(x, cv_var_exp, label="Validation", c=eigen1colour, linestyle='--', linewidth=2.5)
    ax.set_xlabel("PC #")
    ax.set_ylabel("Explained variance")
    sns.despine(ax=ax)
    ax.set_ylim(0, None)

def componentplot(tss, n, ax=plt.gca(), scale="raw"):
    pca = PCA(n_components=n)
    scores = pca.fit_transform(tss)
    ax.set_prop_cycle(plt.cycler(color=sns.color_palette("Set1")))
    if scale == "raw":
        ax.plot(pca.components_.T * pca.explained_variance_ratio_.T)
    elif scale == "log":
        ax.plot(pca.components_.T * np.sqrt(pca.explained_variance_ratio_.T))
    ax.set_yticks([])
    ax.set_ylabel("PC loading")
    sns.despine(ax=ax, left=True)
    return pca.components_, scores

# def trajplot(tss, ax=plt.axes(projection='3d')):
#     pca = PCA(n_components=5)
#     pca.fit(tss)
#     ax.set_prop_cycle(plt.cycler(color=sns.color_palette("Set1")))
#     ax.plot3D(pca.components_[1] * pca.explained_variance_ratio_[1], pca.components_[0] * pca.explained_variance_ratio_[0], pca.components_[2] * pca.explained_variance_ratio_[2])
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])
#     ax.set_xlabel("PC2")
#     ax.set_ylabel("PC1")
#     ax.set_zlabel("PC3")
#     sns.despine(ax=ax, left=True)

def timeseriesplot(tss, n, ax=plt.gca()):
    ax.set_prop_cycle(plt.cycler(color=sns.color_palette("Set2")))
    ax.plot(np.asarray(tss[0:n]).T)
    ax.set_yticks([])
    sns.despine(ax=ax, left=True)
    ax.set_ylabel("Timeseries value")

def scree_legend(c, axname, shift=0, heldout=True):
    c.add_legend(Point(.4+shift, .9, "axis_"+axname), [("Data", {"c": eigen1colour, "linestyle": "-", "linewidth":1.5})] + 
                                                     ([("Held-out data", {"c": eigen1colour, "linestyle": "--", "linewidth": 2.5})] if heldout else []) + 
                                                      [("White noise", {"c": "gray", "linestyle": "-", "linewidth":2, "alpha": .7})],
                 line_spacing=Vector(0, 1.3, "Msize"),
                 sym_width=Vector(2, 0, "Msize"),
                 padding_sep=Vector(.5, 0, "Msize"))

def component_legend(c, axname, n=5):
    items = [(f"PC {i+1}", {"c": sns.color_palette()[i]}) for i in range(0, n)]
    c.add_legend(Point(.5, .9, "axis_"+axname), items,
                 line_spacing=Vector(0, 1.3, "Msize"),
                 sym_width=Vector(2, 0, "Msize"),
                 padding_sep=Vector(.5, 0, "Msize"))

def section_label(c, axname, text, offset=Vector(0,0)):
    c.add_text(text, Point(0, 1, f"axis_{axname}")+Vector(-.1, .25, "in")+offset, weight="bold", ha="left")

def plot_title(c, axname, text):
    try:
        c.add_text(text, Point(.5, 1, f"axis_{axname}")+Vector(0, .30, "in"), ha="center")
    except ValueError:
        c.add_text(text, Point(.5, 1, axname)+Vector(0, .30, "in"), ha="center")

def image_title(c, axname, text):
    try:
        c.add_text(text, Point(.5, 1, f"axis_{axname}")+Vector(0, .10, "in"), ha="center")
    except ValueError:
        c.add_text(text, Point(.5, 1, axname)+Vector(0, .10, "in"), ha="center")
