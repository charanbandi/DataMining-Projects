import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def read_file(filename_string):
    image_list = []
    for line in open(filename_string):
        image_list.append(line[:-1].split(','))
    return image_list


def cluster_func(image_np, k_value):
    # print(image_np.shape)  # (10000, 784)
    sel_random = np.random.permutation(len(image_np))[
                 :k_value]  # select k number of centroids randomly with numpy random permutations
    print(sel_random)
    print(sel_random.shape)
    sel_center = image_np[sel_random]  # pick and assign k number of centers from the random permutation

    while True:
        # 3rd party libraries allowed by Prof to measure distance, mentioned on piazza  #   https://piazza.com/class/kcqkfy8sk3k1s?cid=174_f1
        calc_distance = distance.cdist(image_np, sel_center,
                                       'cosine')  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
        print(calc_distance)
        mindist_list = []
        for x in calc_distance:
            mindist_list.append(
                np.where(x == x.min())[0])  # np where returns the index of the min distance in available vals
        print(mindist_list)

        cluster_id = np.asarray(mindist_list)
        print(cluster_id)
        print(cluster_id.shape)
        cluster_id = cluster_id.reshape(1, 10000)

        for x in range(k_value):
            print(image_np[cluster_id[0] == x].mean(0))

        # get the best fitting center from the mean value of the previous clusters in ranfe of given k
        sel_center_update = np.array([image_np[cluster_id[0] == x].mean(0) for x in range(k_value)])

        # if the selected centroid are clusteres the same way as last loop, break and send data
        if np.all(sel_center == sel_center_update):
            break
        else:
            sel_center = sel_center_update

    print("id's" + str(cluster_id) + "head" + str(cluster_id.shape))
    return sel_center, cluster_id


def output(data, cluster_id, tsne_results):
    f = open("image.txt", "w")
    for x in cluster_id:
        for y in x:
            f.write(str(int(y + 1)) + "\n")
    f.close()

    tsne_df = pd.DataFrame({'X': tsne_results[:, 0],
                            'Y': tsne_results[:, 1],
                            'digit': y})
    sns.scatterplot(x="X", y="Y",
                    hue="digit",
                    palette="pastel",
                    legend='full',
                    data=tsne_df);
    plt.savefig("image_cluster.png")

data = read_file("test_data_image.txt")

k_value = 10  # change variable value to adjust the number of clusters
pca = PCA(n_components=30)
pca_result = pca.fit_transform(data)
tsne = TSNE(perplexity=50, n_iter=2000, n_iter_without_progress=1000, early_exaggeration=4)
tsne_results = tsne.fit_transform(pca_result)

sel_center, cluster_id = cluster_func(np.asarray(tsne_results).astype(np.float), k_value)
output(data, cluster_id,tsne_results)
