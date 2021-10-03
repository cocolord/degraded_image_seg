from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import offsetbox


if __name__ == "__main__":
    labels = np.load("/home/juan/Donglusen/Workspace/mmsegmentation/tests/test_label.npy")
    print(labels.shape)
    color = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    X = TSNE(n_components=2,random_state=33).fit_transform(labels) #(H*W, 19)
    fig = plt.figure(figsize=(64, 64))
    # for i, c, label in zip(target_ids, colors, digits.target_names):
    #     plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    plt.scatter(X[:, 0], X[:, 1], c='r', cmap=plt.cm.Spectral)
    plt.legend()
    plt.savefig("/home/juan/Donglusen/Workspace/mmsegmentation/tests/scatterplot_1.png")