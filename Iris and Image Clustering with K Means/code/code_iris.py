import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance  # https://docs.scipy.org/doc/scipy/reference/spatial.distance.html


def read_file(filename_string):
    iris_list = []
    for line in open(filename_string):
        line = line.strip('\n')  # Remove trailing newline char '\n' at the end of each line
        if line != '':
            slen, swid, plen, pwid = line.split(' ')  # split on space for each slen,swid,plen,pwid
            iris = float(slen), float(swid), float(plen), float(pwid)  # convert string values to float and use an
            # print(iris)
            iris_list.append(iris)  # add into iris list
    # print(iris_list)
    return iris_list


def cluster_func(data, k_value):
    iris_np = np.array(data)  # convert data into numPy array for calculations
    # print(iris_np.shape)      #   (150, 4)

    sel_random = np.random.permutation(len(data))[
                 :k_value]  # select k number of centroids randomly with numpy random permutations
    sel_center = iris_np[sel_random]  # pick and assign k number of centers from the random permutation
    while True:
        # 3rd party libraries allowed by Prof to measure distance, mentioned on piazza  #   https://piazza.com/class/kcqkfy8sk3k1s?cid=174_f1
        calc_distance = distance.cdist(iris_np, sel_center,
                                       'cosine')  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
        # print(calc_distance)
        mindist_list = []
        for x in calc_distance:
            mindist_list.append(
                np.where(x == x.min())[0][0])  # np where returns the index of the min distance in available vals
        # print(mindist_list)

        cluster_id = np.asarray(mindist_list)

        # for x in range(k_value):
        #     print(iris_np[cluster_id == x].mean(0))

        # get the best fitting center from the mean value of the previous clusters in ranfe of given k
        sel_center_update = np.array([iris_np[cluster_id == x].mean(0) for x in range(k_value)])

        # if the selected centroid are clusteres the same way as last loop, break and send data
        if np.all(sel_center == sel_center_update):
            break
        else:
            sel_center = sel_center_update
    return sel_center, cluster_id


def output(data, cluster_id):
    f = open("iris.txt", "w")
    [f.write(str(x + 1) + "\n") for x in cluster_id]
    f.close()

    plt.scatter(np.array(data)[:, 0], np.array(data)[:, 1], c=cluster_id, s=80, cmap='plasma')
    plt.savefig("iris_cluster.png", bbox_inches='tight')


data = read_file("test_data_iris.txt")
k_value = 3  # change variable value to adjust the number of clusters
sel_center, cluster_id = cluster_func(data, k_value)
output(data, cluster_id)
