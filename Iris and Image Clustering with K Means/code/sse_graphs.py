from sklearn. cluster import KMeans
import matplotlib.pyplot as plt

# https://intellipaat.com/community/1867/scikit-learn-k-means-elbow-criterion
# https://www.scikit-yb.org/en/latest/api/cluster/elbow.html

def read_file_iris(filename_string):
    iris_list = []
    for line in open(filename_string):
        line = line.strip('\n') #   Remove trailing newline char '\n' at the end of each line
        if line != '':
            slen,swid,plen,pwid = line.split(' ') # split on space for each slen,swid,plen,pwid
            iris = float(slen),float(swid),float(plen),float(pwid)  #   convert string values to float and use an
            iris_list.append(iris)  # add into iris list
    return iris_list

def read_file_image(filename_string):
    image_list=[]
    for line in open(filename_string):
        image_list.append(line[:-1].split(','))
    return image_list

def graph_sse(data_iris, range_cluster, save_filename):
    sse = {}
    for k in range(1, range_cluster):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data_iris)
        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center

    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.savefig(save_filename, bbox_inches='tight')


data_iris = read_file_iris("test_data_iris.txt")
graph_sse(data_iris,20,"sse_iris.png")

data_image = read_file_image("test_data_image.txt")
graph_sse(data_image,25,"sse_image.png")





