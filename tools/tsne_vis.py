from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import offsetbox

def fashion_scatter(x, colors):
    num_classes = len(np.unique(colors))
    print("NUM CLASSES: " + str(num_classes))
    palette = np.array(sns.color_palette("hls", num_classes))

    #create scatter plot
    f = plt.figure(figsize=(FIGSIZEX, FIGSIZEY))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=200, c=palette[colors.astype(np.int)])  #s= 500 per punti più grandi
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    #ax.axis('off')
    ax.axis('tight')
    plt.savefig("/home/juan/Donglusen/Workspace/mmsegmentation/tests/scatterplot.png")
    

def plot_embedding(X, title=None):
    y = range(0,20)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.savefig("/home/juan/Donglusen/Workspace/mmsegmentation/tests/scatterplot.png")

def rand_int():
    import random
    return random.randint(1,10)

def my_plot_embedding(X):
    labels = np.zeros(len(X),)
    for i in range(len(labels)):
        labels[i] = rand_int()
    X_embedded = TSNE(n_components=2).fit_transform(X)
    print(X_embedded.shape)
    plt.figure()
    plt.scatter(X_embedded[:,0],X_embedded[:,1],c=labels, s=0.5, alpha = 0.5)
    plt.savefig("/home/juan/Donglusen/Workspace/mmsegmentation/tests/scatterplot.png")

if __name__ == "__main__":
    x = np.load("/home/juan/Donglusen/Workspace/mmsegmentation/tests/test_tsne.npy")
    print(x.shape)
    # x = x.reshape(-1, x.shape[1])
    print(x.shape)
    tsne = TSNE()
    x_embedded = tsne.fit_transform(x)
    FIG_FUNC = 'mine'
    if FIG_FUNC == 'fashion':
        sns.set(rc={'figure.figsize':(11.7,8.27)})  
        palette = sns.color_palette("bright", 19)
        fig = sns.scatterplot(x_embedded[:,0],x_embedded[:,1], legend='full',palette="Set1")
        scatter_fig = fig.get_figure()
        scatter_fig.savefig("/home/juan/Donglusen/Workspace/mmsegmentation/tests/scatterplot.png", dpi=400)
    elif FIG_FUNC == 'seaborn':    
        plot_embedding(x_embedded)
    elif FIG_FUNC == 'mine':
        my_plot_embedding(x)
    else:
        colors = [l for l in range(0,20)]
        colors = np.asarray(colors, dtype=np.uint8)
        fashion_scatter(x_embedded, colors)