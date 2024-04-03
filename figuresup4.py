import numpy as np
import networkx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

REVERSE = False

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica Neue LT Std'],'size': 8})

# Generate the graph as an edgelist
np.random.seed(1)
N = 1000
ends = [1]
tree = [(0,1)]
current_node = 2
for i in range(2,N):
    from_end = ends.pop(np.random.randint(0, len(ends)))
    if np.random.rand()>.990:
        ends.append(current_node)
        tree.append((from_end,current_node))
        current_node += 1
    ends.append(current_node)
    tree.append((from_end,current_node))
    current_node += 1

# Use networkx to convert to a graph object and get the positions to draw it
# nicely.
G = networkx.from_edgelist(tree)
pos = networkx.kamada_kawai_layout(G)
posx = np.asarray([pos[i][0] for i in range(0, len(G))])
posy = np.asarray([pos[i][1] for i in range(0, len(G))])
# networkx.draw_networkx(G, pos=pos)
# plt.show()

# Find the shortest path length matrix
shortest_path_dict = dict(networkx.shortest_paths.shortest_path_length(G))
shortest_path_matrix = np.asarray([[shortest_path_dict[i][j] for i in range(0, len(G))] for j in range(0, len(G))])

# For the sorting of the covariance matrix, walk the graph and backtrack.
branchpoints = []
o = [0]
nextedge = tree[0]
for _ in range(0, len(G)-1):
    o.append(nextedge[1])
    nexts = [e for e in tree if e[0] == nextedge[1]]
    if len(nexts) == 0:
        if len(branchpoints) == 0:
            break
        nextedge = branchpoints.pop()
    else:
        nextedge = nexts.pop()
        branchpoints.extend(nexts)

# Get the covariance matrix and sort it
cov = np.exp(-shortest_path_matrix/100)[o][:,o]

# Generate timeseries with this covariance matrix
tss = np.real(np.random.multivariate_normal(np.zeros(len(G)), cov, 50000))
tss_cv = np.real(np.random.multivariate_normal(np.zeros(len(G)), cov, 50000))
# sqrtcov = np.real(scipy.linalg.sqrtm(cov))
# tss = np.random.randn(10000,len(G)) @ sqrtcov

pca = PCA(n_components=12)
pca.fit(tss)

# Note this is slightly different than the one used in the rest of the paper
# since it goes to grey in the middle instead of white
cmap = plt.cm.colors.LinearSegmentedColormap.from_list('grad', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:000000-50:FFFFFF-100:266D9C
    (0.000, (0.000, 0.000, 0.000)),
    (0.500, (0.800, 0.800, 0.800)),
    (1.000, (0.149, 0.427, 0.612) if not REVERSE else (.937, .231, .173))))

cmap.set_over((0, 0, 0))

from cand import Canvas, Point, Vector

c = Canvas(7.0, 2.1, "in")
c.set_font("Nimbus Sans", size=8, ticksize=7)
c.add_axis("cov", Point(.2, .5, "in"), Point(1.4, 1.7, "in"))
c.add_grid([f"PC{i}" for i in range(1, 13)], 2, Point(1.9, .5, "in"), Point(5.2, 1.7, "in"), spacing=Vector(.1, .1, "in"))
c.add_axis("scree", Point(5.9, .5, "in"), Point(6.9, 1.7, "in"))

for i in range(0, 12):
    ax = c.ax(f"PC{i+1}")
    ax.scatter(posx[o], posy[o], c=pca.components_[i], cmap=cmap, s=3)
    c.add_text(f"PC{i+1}", Point(np.min(posx[o])-.3, np.max(posy[o]), f"PC{i+1}"), size=6)
    ax.axis("off")

c.add_text("PC loadings (plotted on the manifold)", (Point(1, 1, "axis_PC3") | Point(0, 1, "axis_PC4")) + Vector(0, .2, "in"))

ax = c.ax("cov")
ax.imshow(cov, cmap="PuOr", vmin=-1, vmax=1)
ax.set_xticks([])
ax.set_yticks([])
c.add_text("Covariance matrix", Point(.5, 1, "axis_cov")+Vector(0, .2, "in"))

from pcalib import screeplot

ax = c.ax("scree")
screeplot(tss[0:1000], tss_cv[0:1000], n=12, ax=ax)
c.add_text("Scree plot", Point(.5, 1, "axis_scree")+Vector(0, .2, "in"))

c.add_figure_labels([("a", "cov"), ("b", "PC1"), ("c", "scree")])

c.save("figuresup4.pdf")
