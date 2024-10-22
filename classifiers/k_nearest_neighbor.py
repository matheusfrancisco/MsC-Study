import math
import numpy as np
import collections
from matplotlib import pyplot as plt

points = {"blue": [[2, 4, 3], [1, 3, 5], [2, 3, 1], [3, 2, 3], [2, 1, 6]],
          "red": [[5, 6, 5], [4, 5, 2], [4, 6, 1], [6, 6, 1], [5, 4, 6]]}


new_point = [3, 3, 4]

#for 2d
#points = {"blue": [[2, 4], [1, 3], [2, 3], [3, 2], [2, 1]],
#          "red": [[5, 6], [4, 5], [4, 6], [6, 6], [5, 4]]}
#
#
#new_point = [3, 3]



def euclidean_distance(p, q):
    # sqrt(sum (px - qx)^2)
    # sqrt(sum ((px - qx)^2 + (py - qy)^2 + ... + (pn - qn)^2))
    return math.sqrt(sum((px - qx) ** 2 for px, qx in zip(p, q)))

def euclidean_distance_np(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

def manhattan_distance(p, q):
    return sum(abs(px - qx) for px, qx in zip(p, q))

class KNearestNeighbor:
    def __init__(self, k):
        self.k = k
        self.point = None

    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []
        for category in self.points:
            for point in self.points[category]:
                d = euclidean_distance(point, new_point)
                #d = manhattan_distance(point, new_point)
                distances.append((d, category))
        categories = [category for d, category in sorted(distances)[:self.k]]
        result = collections.Counter(categories).most_common(1)[0][0]
        return result

knn = KNearestNeighbor(3)
knn.fit(points)
## visualize 

## change to true for 2d
if False:
    ax = plt.subplot()
    ax.grid(True, color="#323232")
    ax.set_facecolor("black")
    ax.figure.set_facecolor("#121212")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    
    for point in points['blue']:
        ax.scatter(point[0], point[1], color="#104DCA", s=60)
    
    for point in points['red']:
        ax.scatter(point[0], point[1], color="#FF0000", s=60)
    
    new_class = knn.predict(new_point)
    color = "#FF0000" if new_class == "red" else "#104DCA"
    ax.scatter(new_point[0], new_point[1], color=color, s=200, zorder=100, marker="x")
    
    
    for point in points['blue']:
        ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color="#104DCA", linestyle="--", linewidth=1)
    
    for point in points['red']:
        ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color="#FF0000", linestyle="--", linewidth=1)
    
    plt.show()
else:
    print(1)
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(projection='3d')
    ax.grid(True, color="#323232")
    ax.set_facecolor("black")
    ax.figure.set_facecolor("#121212")
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")
    
    for point in points['blue']:
        ax.scatter(point[0], point[1], point[2], color="#104DCA", s=60)
    
    for point in points['red']:
        ax.scatter(point[0], point[1], point[2], color="#FF0000", s=60)
    
    new_class = knn.predict(new_point)
    color = "#FF0000" if new_class == "red" else "#104DCA"
    ax.scatter(new_point[0], new_point[1], new_point[2], color=color, s=200, zorder=100, marker="x")
    
    
    for point in points['blue']:
        ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]], color="#104DCA", linestyle="--", linewidth=1)
    
    for point in points['red']:
        ax.plot([new_point[0], point[0]], [new_point[1], point[1]], [new_point[2], point[2]], color="#FF0000", linestyle="--", linewidth=1)
    
    plt.show()
