import numpy
import logging
from numpy import zeros, logical_or, nonzero, amax, mat, vstack

dlog = logging.getLogger('data')
ilog = logging.getLogger('info')
numpy.set_printoptions(precision=4, linewidth=10000, suppress=True)

class Classifier(object):
    def train(self, data, labels):
        raise NotImplementedError()

    def predict(self, data, mode):
        raise NotImplementedError()

    def cross_validation(self, data, labels):
        raise NotImplementedError()

    def examine(self, prediction, target):
        """
        Compare predicted and correct results
        prediction: N-by-K matrix, predicted results. The first dimension is the number of data points, the second dimension is the number of classes.
        target: N-by-K matrix, predicted results. The first dimension is the number of data points, the second dimension is the number of classes.
        """
        N1, K1 = prediction.shape
        N2, K2 = target.shape
        if N1!=N2 or K1!=K2:
            raise Exception('prediction and target dimensions do not match.')

        err_count = 0
        mis_classified = []
        for i in range(N1):
            if (prediction[i] != target[i]).any():
                err_count+=1
                mis_classified.append(i)

        return err_count*1.0/N1, mis_classified

    def evaluate(self, test_data, test_labels):
        N, K = test_labels.shape
        Z_t_b, Z_t_r = self.predict(test_data)
        true_positives   = zeros(K)
        true_negatives   = zeros(K)
        false_positives  = zeros(K)
        false_negatives  = zeros(K)
        total_positive_count = zeros(K)
        wrong_cases = []

        for i in range(N):
            l = numpy.array(test_labels[i]==1)
            p = numpy.array(Z_t_b[i]==1)

            total_positive_count[p]+=1
            if (p==l).all() == True:
                true_positives[l]+=1
                true_negatives[l==False]+=1
            else:
                wrong_cases.append(i+1)
                false_positives[p]+=1
                false_negatives[l]+=1
                true_negatives[logical_or(p,l)==False]+=1

        total_negative_count = N - total_positive_count

        stats_tag="[Stats]"
        dlog.debug("%s True positives:  [%s]" % (stats_tag, true_positives))
        dlog.debug("%s False positives: [%s]" % (stats_tag, false_positives))
        dlog.debug("%s True negatives:  [%s]" % (stats_tag, true_negatives))
        dlog.debug("%s False negatives: [%s]" % (stats_tag, false_negatives))

        dlog.debug("%s True positives rate:  [%s]" % (stats_tag, true_positives/total_positive_count))
        dlog.debug("%s False positives rate: [%s]" % (stats_tag, false_positives/total_positive_count))
        dlog.debug("%s True negatives rate:  [%s]" % (stats_tag, true_negatives/total_negative_count))
        dlog.debug("%s False negatives rate: [%s]" % (stats_tag, false_negatives/total_negative_count))

        wrong_tag="[Wrong]"
        dlog.debug("%s %s" % (wrong_tag, wrong_cases))

        detail_tag="[Details]"
        for i in range(N):
            ind = i+1
            predict = nonzero(Z_t_b[i]==1)[0][0]
            actual  = nonzero(test_labels[i]==1)[0][0]
            mark = "Correct" if ind not in wrong_cases else "Incorrect"
            dlog.debug("%s[%d][%s][Predict: %d][Actual: %d][Highest Score: %s] %s\n" % (detail_tag, ind, mark, predict, actual, amax(Z_t_r[i]), Z_t_r[i]))


    def to_vector_labels(self, labels, num_classes):
        """
        Variable labels is a N-by-1 metrix of the labels of N data points, the value should be 0-based continous integers
        num_classes indicate the total number of different classes
        e.g. label 3 in a 10-class classification is translated to [0,0,0,1,0,0,0,0,0,0]
        """
        expanded = []
        N, K = labels.shape
        for i in range(N):
            vector = zeros((1,num_classes))
            vector[0, labels[i,0]] = 1
            expanded.append(vector)
        expanded = vstack(expanded)
        return expanded

    def to_scalar_labels(self, labels):
        """
        """
        N, K = labels.shape
        scalar_labels=[]
        for i in range(N):
            label = nonzero(labels[i]==1)[1][0,0]
            scalar_labels.append(label)
        return mat(scalar_labels)

if __name__ == "__main__":
    c = Classifier()
    print c.to_vector_labels(mat([0, 1,2]).T, 3)
    print c.to_scalar_labels(mat([[1,0,0],[0,1,0],[0,0,1]]))