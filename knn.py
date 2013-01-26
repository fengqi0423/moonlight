import numpy
from numpy import multiply, mat
from base import Classifier

class KNNClassifier(Classifier):
    def __init__(self, k):
        self.k = k

    def train(self, data, labels):
        self.neighbors = data
        self.labels = labels

    def predict(self, data):
        predictions = []
        for datum in data:
            neighbors = self.find_neigbhors_for_one(datum)
            neighbor_labels = [n[1] for n in neighbors]
            distances = [n[0] for n in neighbors]
            print neighbors
            p = self.decide(distances, neighbor_labels)
            predictions.append(p)
        return predictions

    def find_neigbhors_for_one(self, datum):
        N, K = self.neighbors.shape
        nearests = [(
            self.distance(self.neighbors[0], datum),
            self.labels[0,0],
            0
        )] # Keep nearest neighbors in ascending order
        count = 1
        for i in range(N):
            d = self.distance(self.neighbors[i], datum)
            for j in range(count):
                if d < nearests[j][0]:
                    nearests.insert(j, (d, self.labels[i, 0], i))
                    count+=1
                    if count > self.k:
                        nearests.pop()
                        count-=1
        return nearests

    def decide(self, distances, neighbor_labels):
        votes  = {0: 0}
        winner = 0
        for l in neighbor_labels:
            if l in votes.keys():
                votes[l]+=1
            else:
                votes[l]=1
            if votes[l] > votes[winner]:
                winner = l
        return winner

    def distance(self, point1, point2):
        diff = point1-point2
        return numpy.sum(multiply(diff, diff))

if __name__ == "__main__":
    knn = KNNClassifier(1)
    train_data = mat([
        [0,0], 
        [0,1],
        [1,0],
        [1,1],
        ])
    train_label = mat([
        [1],
        [0],
        [0],
        [1],
        ])
    knn.train(train_data, train_label)
    print knn.predict(train_data)

