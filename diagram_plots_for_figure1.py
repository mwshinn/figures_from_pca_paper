import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
import sklearn.decomposition
from sklearn.decomposition import PCA

purple = "#beaed4ff"
score1colour = "#ef3b2cff"
score2colour = "#fb6a4aff"
score3colour = "#fc9272ff"
score4colour = "#fcbba1ff"
scorecolours = [score1colour, score2colour, score3colour, score4colour]

loading1colour = "#2171b5ff"
loading2colour = "#4292c6ff"
loading3colour = "#6baed6ff"
loading4colour = "#9ecae1ff"
loadingcolours = [loading1colour, loading2colour, loading3colour, loading4colour]

eigen1colour = "#238b45ff"
eigen2colour = "#41ab5dff"
eigen3colour = "#74c476ff"
eigen4colour = "#a1d99bff"


f = plt.figure(figsize=(2,2))
ax = f.gca()
N = 200
# points1 = np.random.randn(N) + np.linspace(0, 3, N)
# points2 = np.random.randn(N) + np.linspace(-10, 20, N)
#ax.scatter(np.linspace(0, 1, N), points1, s=20, color=purple, edgecolors='k', linewidths=.3)
#ax.scatter(np.linspace(0, 1, N), points2, s=20, color=purple, edgecolors='k', linewidths=.3)
# rn1 = np.tile(np.linspace(0, 1, N),N)#np.random.rand(N)
# rn2 = np.repeat(np.linspace(0, 1, N),N)#np.random.rand(N)
np.random.seed(0)
rn1 = np.random.rand(N)
rn2 = np.random.rand(N)
# rn1 = np.tile(np.linspace(0, 1, N),N) + np.random.randn(N*N)*.04
# rn2 = np.repeat(np.linspace(0, 1, N),N) + np.random.randn(N*N)*.04
ab1 = rn1+.7*rn2
ab2 = rn2 + .4
ax.scatter(ab1, ab2, s=20, color=purple, edgecolors='k', linewidths=.3, clip_on=False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-.1, 1.7)
ax.set_ylim(-.1, 1.7)
sns.despine(ax=ax)
plt.savefig("diagram1-bias1.svg")
plt.show()


f = plt.figure(figsize=(2,2))
ax = f.gca()
N = 80
points1 = np.random.randn(N) + 7*np.cos(np.linspace(0, 5, N))
ax.scatter(np.linspace(0, 1, N), points1, s=20, color=purple, edgecolors='k', linewidths=.3)
ax.set_xticks([])
ax.set_yticks([])
sns.despine(ax=ax)
plt.savefig("diagram1-bias2.svg")
plt.show()


np.random.seed(2)
N = 30
points = scipy.ndimage.gaussian_filter1d(np.random.randn(300000, N*5), 4)[:,N*2:(4*N)]

# Smooth timeseries
f = plt.figure(figsize=(2,2))
ax = f.gca()
ax.plot(points[300:304].T-np.mean(points[300:304].T, axis=0, keepdims=True), color=purple, linewidth=3)
ax.set_xticks([])
ax.set_yticks([])
sns.despine(ax=ax)
plt.savefig("diagram1-bias3.svg")
plt.show()

# More smooth timeseries
f = plt.figure(figsize=(2,2))
ax = f.gca()
ax.plot(points[200:204].T-np.mean(points[200:204].T, axis=0, keepdims=True), color=purple, linewidth=3)
ax.set_xticks([])
ax.set_yticks([])
sns.despine(ax=ax)
plt.savefig("diagram1-bias3-ex2.svg")
plt.show()



pca = PCA(n_components=4)
pca.fit(points)


f = plt.figure(figsize=(2,2))
ax = f.gca()
for i in range(0, len(pca.explained_variance_ratio_)):
    plt.plot(pca.components_.T[:,i]*pca.explained_variance_ratio_[None,i]*((pca.components_[:,0]>0)*2-1)[i], c=loadingcolours[i], linewidth=3)

ax.set_xticks([])
ax.set_yticks([])
sns.despine(ax=ax)
plt.savefig("diagram1-pc-coss.svg")
plt.show()

f = plt.figure(figsize=(2,2))
ax = f.gca()
for i in range(0, len(pca.explained_variance_ratio_)):
    plt.plot(pca.components_.T[:,i]*pca.explained_variance_ratio_[None,i]*((pca.components_[:,0]>0)*2-1)[i], c=scorecolours[i], linewidth=3)

ax.set_xticks([])
ax.set_yticks([])
sns.despine(ax=ax)
plt.savefig("diagram1-scores-coss.svg")
plt.show()



f = plt.figure(figsize=(2,2))
ax = f.gca()
N = 5
np.random.seed(0)
curve = scipy.stats.norm.pdf(np.linspace(-10, 10, 201))
shifts = ((np.random.rand(N)-.5)*90).astype(int)
curves = np.asarray([curve[(50+s):(150+s)] for s in shifts])
ax.plot(curves.T, color=purple, linewidth=3)
ax.set_xticks([])
ax.set_yticks([])
sns.despine(ax=ax)
plt.savefig("diagram1-bias4.svg")
plt.show()

lots_of_shifts = ((np.random.rand(10000)-.5)*30).astype(int)
lots_of_curves = np.asarray([curve[(50+s):(150+s)] for s in lots_of_shifts])
pca = sklearn.decomposition.PCA(n_components=5)
pca.fit(lots_of_curves)

f = plt.figure(figsize=(2,2))
ax = f.gca()
for i in range(0, 4):
    ax.plot(pca.components_[i]*np.sqrt(1*pca.explained_variance_ratio_[i]), color=loadingcolours[i], linewidth=3)

ax.set_xticks([])
ax.set_yticks([])
sns.despine(ax=ax)
plt.savefig("diagram1-bias4-pcs.svg")
plt.show()


