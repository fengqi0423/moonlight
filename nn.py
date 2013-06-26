import numpy
import logging
import pylab
from numpy import ones, zeros, mat, hstack, vstack, mean, std, multiply, square, argmax, amax, transpose, logical_or, nonzero
from matplotlib import pyplot as plt
from base import Classifier

dlog = logging.getLogger('data')
ilog = logging.getLogger('info')
dlog.setLevel(logging.DEBUG)
ilog.setLevel(logging.DEBUG)
dlog.addHandler(logging.FileHandler('test.log'))
ilog.addHandler(logging.StreamHandler())
numpy.set_printoptions(precision=4, linewidth=10000, suppress=True)

class NeuralNetworkClassifier(Classifier):
    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=0.01, train_epoches=5, error_tolerance=0, momentum=0.5):
        """
        input_dim  - input dimension
        output_dim - output dimension
        hidden_dim - number of hidden units, per layer, in the order from input to output
        learning_rate - the numerical learning_rate for the stochastic gradient descend. Note the training method automatically normalize data
        train_epoches - the number of times the entire training set is used
        error_tolerance - target mis-classification rate, if achieved in training data, training will stop.
        """
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim if type(hidden_dim)==list else [hidden_dim]
        self.learning_rate = learning_rate
        self.train_epoches = train_epoches
        self.error_tolerance = error_tolerance
        self.momentum = momentum

        # Initialize weights layers
        self.weights = []
        for i in range(len(self.hidden_dim)):
            if i == 0:
                w = mat(numpy.random.random((self.input_dim+1,  self.hidden_dim[i])))
            else:
                w = mat(numpy.random.random((self.hidden_dim[i-1]+1, self.hidden_dim[i])))
            self.weights.append(w)
        self.weights.append(mat(numpy.random.random((self.hidden_dim[-1]+1,  self.output_dim))))
        self.nlayers = len(self.weights)

    def normalize(self, A):
        """
        A should be a matrix. The first dimension represents data points, and the second represent the data space.
        """
        S = std(A, 0)
        M = mean(A, 0)

        N, K = A.shape
        zero_std_dims  = [i for i in range(K) if S[0,i]<=1]
        ilog.warn("WARNING: Dimension: %s has ZERO standard deviation. You should remove them from dataset." % str(zero_std_dims))

        for dim in zero_std_dims:
            S[0,dim] = 1
        
        return (A-M)/S, M, S

    def save_weights(self, file_name='weights.txt'):
        with open(file_name, 'w') as f:
            for i in range(self.nlayers):
                w = self.weights[i]
                a, b = w.shape
                f.write("%d|%d|%d\n" % (i, a, b))
                a1 = numpy.array(w)
                for i in range(a):
                    l = ','.join([str(val) for val in a1[i]])
                    f.write(l+"\n")

    def load_weights(self, file_name='weights.txt'):
        with open(file_name) as f:
            self.weights = []
            self.hidden_dim = []
            self.nlayers = 0
            l = f.readline()
            while l:
                _,a,b = [int(val) for val in l.strip('\n').split('|')]
                self.nlayers += 1
                self.hidden_dim.append(a)
                w = []
                for i in range(a):
                    l = f.readline()
                    w.append([float(val) for val in l.split(',')])
                self.weights.append(vstack(w))
                l = f.readline()

    def activate(self, x):
        """
        Apply the activation function - this is for constraining the ouput to between 0 and 1. 
        The activation function should be differentiable everywhere. For now I only support sigmoid function.
        x can be matrix or vector or scalar
        """
        y = 1/(1+numpy.exp(-x))
        y[(x < -50)] = 0
        y[(x > 50)]  = 1
        return y

    def normalized_activate(self, A):
        """
        Apply the activation function to multiclass output - this should make sure the probability for all output units sum up to one.
        For training I still use the activate method - just to simplify the problem.
        I only support softmax function now.
        x should be a matrix. The first dimension represent the number of data points, and the second dimension represents the output space.
        """
        Z = numpy.exp(A)
        S = sum(Z, 1)
        return Z/S # omg numpy is smart!

    def predict(self, data, mode='binary'):
        """
        data - N-by-M matrix, training data input. The first dimension is data points, and the second dimension is input space.
        mode - binary: return yes/no prediction
               raw: return raw probabilities
               normalized: return posterior/normalized probabilities
        """
        N, M = data.shape
        if M != self.input_dim:
            raise Exception("Wrong dimension for training data -  expecting %d, seeing %d" % (self.input_dim, M))

        Y = data
        L = len(self.weights)
        NValues = [] # Need to store each neuron's value, 
        for i in range(L):
            w = self.weights[i]
            N, M = Y.shape
            os = ones((N,1), numpy.float)        # Padding ones to support the offset parameter
            Y  = hstack((Y, os))
            NValues.append(Y)       # This will store every layer's output except the real output neurons'
            Y  = Y*w
            if i < L-1:
                Y  = self.activate(Y)
    
        if mode=='raw':
            Z = self.activate(Y)
        elif mode=='normalized':
            Z = self.normalized_activate(Y)
        elif mode=='binary':
            M = numpy.amax(Y, 1)
            Z = zeros(Y.shape)
            Z[(Y==M)]=1

        return Z, NValues

    def train(self, data, labels, test_data=None, test_labels=None):
        """
        data  - N-by-M matrix, training data input. The first dimension is data points, and the second dimension is input space.
        labels - N-by-K matrix, training data output. The first dimension is data points, and the second dimension is output space.
        test_data - J-by-M matrix, testing data input. This is just for reportiing error rate during the training process.
        test_labels - J-by-K matrix, testing data output. The first dimension is data points, and the second dimension is output space.
        Training uses stochastic gradient descend with backpropagation.
        """
        if test_data is None or test_labels is None:
            test_data = data
            test_labels = labels

        N1, L = data.shape
        N2, K = labels.shape
        if L != self.input_dim:
            raise Exception("Wrong dimension for training data - expecting %d, seeing %d" % (self.input_dim, L))
        if K != self.output_dim:
            raise Exception("Wrong dimension for training labels - expecting %d, seeing %d" % (self.output_dim, K))
        if N1 != N2:
            raise Exception("Input and output dimensions do not match")


        normalized_data, M, S = self.normalize(vstack((data, test_data)))
        data      = normalized_data[:N1]
        test_data = normalized_data[N1:]

        report_block  = max(1, N1/20)
        learning_rate = self.learning_rate
        momentum      = self.momentum

        layers = range(self.nlayers)
        layers.reverse()
        previous_dw = [None] * self.nlayers
        error_rates = []
        for i in range(self.train_epoches):

            ilog.debug("Start training epoch %d" % i)
            # Since I decided to use stochastic gradient method, here is the interation over data points

            for j in range(N1):
                datum = data[j] # Geeee it's still a matrix!
                label = labels[j]

                # Multi-task learning essentially makes the training data for each class very imbalanced
                # One out of K data points is positive for each class.
                # So amplify this positive training point by K-1 times
                z, nvalues = self.predict(datum, 'raw')

                # Now compute the gradient for each layer
                # Start from the top layer
                signal = label - z
                assert len(layers) == len(nvalues), Exception("There are %d weight layers with only %d input layers" % (len(layers), len(nvalues)))

                for l in layers:
                    weight = self.weights[l]
                    input_val = nvalues[l]

                    if l < self.nlayers - 1:
                        # Except for the output layer, 
                        # the offset neurons are not connected to lower layers
                        signal = signal[:,:-1]

                    dw = transpose(input_val) * signal
                    signal = signal * transpose(weight) 
                    signal = multiply(signal, nvalues[l])
                    signal = multiply(signal, 1-nvalues[l])

                    if previous_dw[l] != None:
                        dw = previous_dw[l]*momentum + (1-momentum)*dw
                    weight = weight + dw * learning_rate

                    self.weights[l] = weight
                    previous_dw[l] = dw

                if j % report_block == 0:
                    ilog.debug("Finished %d%%" % (j*100/N1))
                    if test_data is not None and test_labels is not None:
                        Z_t, _ = self.predict(test_data, 'binary')
                        err_rate, err_cases = self.examine(Z_t, test_labels)
                        error_rates.append(err_rate)
                        ilog.debug("Error rate: %f" % err_rate)
                        if err_rate <= self.error_tolerance: 
                            self.evaluate(test_data, test_labels)
                            # De-normalize weights
                            self.weights[0][L] = self.weights[0][L]-multiply(M, 1/S)*self.weights[0][0:L]
                            for i in range(L):
                                self.weights[0][i] = self.weights[0][i]/S[0,i]
                            plt.plot(range(len(error_rates)), error_rates)
                            pylab.show()
                            return

        self.evaluate(test_data, test_labels)
        # De-normalize weights
        self.weights[0][L] = self.weights[0][L]-multiply(M, 1/S)*self.weights[0][0:L]
        for i in range(L):
            self.weights[0][i] = self.weights[0][i]/S[0,i]
        plt.plot(range(len(error_rates)), error_rates)
        pylab.show()
        return


    def evaluate(self, test_data, test_labels):
        N, K = test_labels.shape
        Z_t_b, _ = self.predict(test_data, 'binary')

        test_labels = numpy.array(test_labels)
        Z_t_b = numpy.array(Z_t_b)
        true_positives   = zeros(K)
        true_negatives   = zeros(K)
        false_positives  = zeros(K)
        false_negatives  = zeros(K)
        total_positive_count = zeros(K)
        wrong_cases = []

        for i in range(N):
            l = test_labels[i]==1
            p = Z_t_b[i]==1
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
        Z_t_r, _ = numpy.array(self.predict(test_data, 'raw'))
        for i in range(N):
            ind = i+1
            predict = nonzero(Z_t_b[i])[0]
            actual  = nonzero(test_labels[i])[0]
            mark = "Correct" if ind not in wrong_cases else "Incorrect"
            dlog.debug("%s[%d][%s][Predict: %d][Actual: %d][Highest Conf: %s] %s\n" % (detail_tag, ind, mark, predict, actual, amax(Z_t_r[i]), Z_t_r[i]))


if __name__ == "__main__":
    nn = NeuralNetworkClassifier(2, 2, [4,3], 2, 1000)
    train_data = mat([
        [9,6], 
        [10,2],
        [5,5],
        [3,1],
        ])
    train_label = mat([
        [1,0],
        [1,0],
        [0,1],
        [0,1],
        ])
    nn.train(train_data, train_label)

    nn.save_weights()
    nn.load_weights()
    print "Reloaded weights from file."
    nn.evaluate(train_data, train_label)
    
