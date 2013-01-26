import numpy
import logging
from numpy import ones, zeros, mat, hstack, vstack, mean, std, multiply, square, argmax, amax
from base import Classifier

dlog = logging.getLogger('data')
ilog = logging.getLogger('info')
numpy.set_printoptions(precision=4, linewidth=10000, suppress=True)

class NeuralNetworkClassifier(Classifier):
    def __init__(self, input_dim, output_dim, hidden_dim, learning_rate=0.01, train_epoches=5, error_tolerance=0, momentum=0.5):
        """
        input_dim  - input dimension
        output_dim - output dimension
        hidden_dim - number of hidden units
        learning_rate - the numerical learning_rate for the stochastic gradient descend. Note the training method automatically normalize data
        train_epoches - the number of times the entire training set is used
        error_tolerance - target mis-classification rate, if achieved in training data, training will stop.
        """
        self.nlayer = 2 # No. of weight layers, only supports 2 now
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.train_epoches = train_epoches
        self.error_tolerance = error_tolerance
        self.momentum = momentum

        # Initialize weights layer
        self.weights_1 = mat(numpy.random.random((self.input_dim+1,  self.hidden_dim)))
        self.weights_2 = mat(numpy.random.random((self.hidden_dim+1, self.output_dim)))

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
            a, b = self.weights_1.shape
            c, d = self.weights_2.shape
            f.write("%d,%d,%d,%d\n" % (a,b,c,d))
            a1 = numpy.array(self.weights_1)
            a2 = numpy.array(self.weights_2)
            for i in range(a):
                l = ','.join([str(val) for val in a1[i]])
                f.write(l+"\n")
            for i in range(c):
                l = ','.join([str(val) for val in a2[i]])
                f.write(l+"\n")

    def load_weights(self, file_name='weights.txt'):
        with open(file_name) as f:
            l = f.readline()
            a,b,c,d = [int(val) for val in l.split(',')]
            self.input_dim  = a-1
            self.output_dim = d
            self.hidden_dim = b
            weights_1 = []
            weights_2 = []
            for i in range(a):
                l = f.readline()
                weights_1.append([float(val) for val in l.split(',')])
            self.weights_1 = vstack(weights_1)
            for i in range(c):
                l = f.readline()
                weights_2.append([float(val) for val in l.split(',')])
            self.weights_2 = vstack(weights_2)

    def activate(self, x):
        """
        Apply the activation function - this is for constraining the ouput to between 0 and 1. 
        The activation function should be differentiable everywhere. For now I only support sigmoid function.
        x can be matrix or vector or scalar
        """
        ill = (x < -50)
        y = 1/(1+numpy.exp(-x))
        y[ill] = 0
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

        os = ones((N,1), numpy.float)        # Padding ones to support the offset parameter
        X  = hstack((data, os))
        Y  = X*self.weights_1
        Y  = self.activate(Y)
        Y  = hstack((Y, os))
        Z  = Y*self.weights_2
        if mode=='raw':
            Z = self.activate(Z)
        elif mode=='normalized':
            Z = self.normalized_activate(Z)
        elif mode=='binary':
            M  = numpy.amax(Z, 1)
            Zr = zeros(Z.shape)
            Zr[(Z==M)]=1
            Z  = Zr

        return X, Y, Z

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
        d_weights_1   = mat(zeros(self.weights_1.shape))
        d_weights_2   = mat(zeros(self.weights_2.shape))
        for i in range(self.train_epoches):

            ilog.debug("Start training epoch %d" % i)
            # Since I decided to use stochastic gradient method, here is the interation over data points
            for j in range(N1):
                datum = data[j] # Geeee it's still a matrix!
                label = labels[j]

                # Multi-task learning essentially makes the training data for each class very imbalanced
                # One out of K data points is positive for each class.
                # So amplify this positive training point by K-1 times
                x, y, z = self.predict(datum, 'raw')

                delta_z = (z-label)
                delta_y = multiply((1-z), z) # delta_y has dimension of z
                delta_y = multiply(delta_y, delta_z)
                delta_x = self.weights_2*delta_y.T
                delta_x = multiply(delta_x.T, y)
                delta_x = multiply(delta_x, 1-y)

                d_weights_2 = y.T*delta_y*(1-momentum) + d_weights_2*momentum
                d_weights_1 = x.T*delta_x[:,0:self.hidden_dim]*(1-momentum) + d_weights_1*momentum

                self.weights_2 = self.weights_2-learning_rate*d_weights_2
                self.weights_1 = self.weights_1-learning_rate*d_weights_1

                if j % report_block == 0:
                    ilog.debug("Finished %d%%" % (j*100/N1))
                    if test_data is not None and test_labels is not None:
                        X_t, Y_t, Z_t = self.predict(test_data, 'binary')
                        err_rate, err_cases = self.examine(Z_t, test_labels)
                        ilog.debug("Error rate: %f" % err_rate)
                        if err_rate <= self.error_tolerance: 
                            # De-normalize weights
                            self.weights_1[L] = self.weights_1[L]-multiply(M, 1/S)*self.weights_1[0:L]
                            for i in range(L):
                                self.weights_1[i] = self.weights_1[i]/S[0,i]
                            return

        self.evaluate(test_data, test_labels)
        # De-normalize weights
        self.weights_1[L] = self.weights_1[L]-multiply(M, 1/S)*self.weights_1[0:L]
        for i in range(L):
            self.weights_1[i] = self.weights_1[i]/S[0,i]
        return

if __name__ == "__main__":
    nn = NeuralNetworkClassifier(2, 1, 2)