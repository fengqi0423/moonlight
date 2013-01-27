import logging
import numpy
from numpy import multiply, mat, vstack, zeros, argmax
from base import Classifier

class KNNClassifier(Classifier):
    def __init__(self, k, output_dim):
        self.k = k
        self.output_dim = output_dim

    def train(self, data, labels):
        self.neighbors = data
        self.labels = labels

    def predict(self, data, mode):
        predictions = []
        for datum in data:
            neighbors = self.find_neigbhors_for_one(datum)
            neighbor_labels = [n[1] for n in neighbors]
            distances = [n[0] for n in neighbors]
            votes, winner = self.poll(distances, neighbor_labels)
            if mode == 'binary':
                predictions.append(winner)
            elif mode == 'raw':
                predictions.append(votes)
        return vstack(predictions)

    def find_neigbhors_for_one(self, datum):
        # Haven't considered equal-distance scenario
        N, K = self.neighbors.shape
        nearests = [(
            self.distance(self.neighbors[0], datum),
            self.labels[0,0],
            0
        )] # Keep nearest neighbors in ascending order
        count = 1
        for i in range(N):
            d = self.distance(self.neighbors[i], datum)
            inserted = False
            for j in range(count):
                if d < nearests[j][0]:
                    nearests.insert(j, (d, self.labels[i, 0], i))
                    count+=1
                    if count > self.k:
                        nearests.pop()
                        count-=1
                    inserted = True
            if not inserted and count < self.k:
                nearests.append((d, self.labels[i, 0], i))
                count+=1
        return nearests

    def poll(self, distances, neighbor_labels):
        """
        Simple count for now. Will use distance data in future.
        """
        votes = mat(zeros((1, self.output_dim)))
        for l in neighbor_labels:
            votes[0, l]+=1
        winner = zeros((1, self.output_dim))
        winner[0, argmax(votes, 1)] = 1
        return votes, winner

    def distance(self, point1, point2):
        diff = point1-point2
        return numpy.sum(multiply(diff, diff))

if __name__ == "__main__":
    logging.getLogger('data').setLevel(logging.DEBUG)
    logging.getLogger('info').setLevel(logging.DEBUG)
    logging.getLogger('data').addHandler(logging.StreamHandler())
    logging.getLogger('info').addHandler(logging.StreamHandler())

    knn = KNNClassifier(3, 2)
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
    test_data = mat([
        [0,0],
        [0,1],
        [1,0],
        [1,1],
        ])    
    test_label = mat([
        [1],
        [0],
        [0],
        [1],
        ])
    knn.train(train_data, train_label)
    print knn.predict(test_data, 'raw')
    print knn.predict(test_data, 'binary')
    knn.evaluate(test_data, knn.to_vector_labels(test_label, 2))

