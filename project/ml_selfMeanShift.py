import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
import random 

X= np.array([[1,2],
            [6.5,1.8],
            [5,8],
            [8,8],
            [1.75,0.6],
##            [9,11],
             [3,4],
                     [7,3],
##                     [9,7],
                     [6,1],
                     [4,9],
                     [2,4],
             [4.5,7]])

##plt.scatter(X[:,0], X[:,1], s=15, linewidths =5)
##plt.show()

colors = 10*["g", "r","c", "b", "k", "o"]


class Mean_Shift:                                                                   ###A LOT OF ROOM FOR IMPROVEMENT.
    def __init__(self, radius = None, radius_norm_step = 100):                  #radius is a major factor in success make it dynamic
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self,data):
        if self.radius == None:
            all_data_centroid = np.average(data,axis =0)
            all_data_norm = np.linalg.norm(all_data_centroid)  ## distance of points from origin
            self.radius = all_data_norm/ self.radius_norm_step
        
        
        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        weights = [i for i in range(self.radius_norm_step)][::-1]   ##from 99 to 0 as weights will reduce with distace

            
        while True:
            new_centroids =[]           ## list to be populated by the new centroids
            for i in centroids:
                in_bandwidth = []   ## all feature sets within our bandwidth
                centroid = centroids[i]

                for featureset in data:
                    distance = np.linalg.norm(featureset - centroid)
                    if distance  == 0:
                        distance = 0.00000001
                    weight_index = int(distance/self.radius)
                    if weight_index > self.radius_norm_step -1:
                        weight_index= self.radius_norm_step -1
                        
                    to_add = (weights[weight_index]**2)*[featureset]
                    in_bandwidth +=to_add

                new_centroid = np.average(in_bandwidth , axis=0)
                new_centroids.append(tuple(new_centroid))  ## as we have to take unique values later and set of tuples works

            uniques = sorted(list(set(new_centroids)))
            to_pop=[]
            for i in uniques:
                for ii in uniques:
                    if i==ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii))<= self.radius:
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i] , prev_centroids[i]):
                    optimized = False

                if not optimized:
                    break
            if optimized:
                break

        self.centroids = centroids
    def predict(self,data):
        pass
clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids
plt.scatter(X[:,0], X[:,1], s=15, linewidths =5)

for c in centroids:
    plt.scatter(centroids[c][0] , centroids[c][1],color= 'k', marker='*',s=150 )
plt.show()
