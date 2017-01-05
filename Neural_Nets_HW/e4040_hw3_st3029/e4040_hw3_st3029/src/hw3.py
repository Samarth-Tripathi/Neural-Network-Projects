"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

Instructor: Prof. Zoran Kostic

This code is based on
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
"""
from __future__ import print_function
import timeit
import inspect
import sys
import scipy 
import numpy

import theano
import theano.tensor as T

from theano.tensor.nnet import conv
from theano.tensor.nnet import conv2d
#from theano.tensor.nnet import relu
from theano.tensor.signal import pool
from theano.tensor.signal import downsample
from hw3_utils import load_data
from hw3_utils import load_data_augmentation
from hw3_utils import shared_dataset


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

#Problem 1
#Implement the convolutional neural network architecture depicted in HW3 problem 1
#Reference code can be found in http://deeplearning.net/tutorial/code/convolutional_mlp.py

class LeNetConvPoolLayer(object):
        """Pool Layer of a convolutional network """

        def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
            """
            Allocate a LeNetConvPoolLayer with shared variable internal parameters.

            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights

            :type input: theano.tensor.dtensor4
            :param input: symbolic image tensor, of shape image_shape

            :type filter_shape: tuple or list of length 4
            :param filter_shape: (number of filters, num input feature maps,
                                  filter height, filter width)

            :type image_shape: tuple or list of length 4
            :param image_shape: (batch size, num input feature maps,
                                 image height, image width)

            :type poolsize: tuple or list of length 2
            :param poolsize: the downsampling (pooling) factor (#rows, #cols)
            """

            assert image_shape[1] == filter_shape[1]
            self.input = input

            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                       numpy.prod(poolsize))
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)

            # convolve input feature maps with filters
            conv_out = conv2d(
                input=input,
                filters=self.W,
                filter_shape=filter_shape,
                input_shape=image_shape
            )

            # pool each feature map individually, using maxpooling
            pooled_out = pool.pool_2d(
                input=conv_out,
                ds=poolsize,
                ignore_border=True
            )

            # add the bias term. Since the bias is a vector (1D array), we first
            # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
            # thus be broadcasted across mini-batches and feature map
            # width & height
            self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

            # store parameters of this layer
            self.params = [self.W, self.b]

            # keep track of model input
            self.input = input

def train_nn(train_model, validate_model, test_model,
                n_train_batches, n_valid_batches, n_test_batches, n_epochs,
                verbose = True):
        """
        Wrapper function for training and test THEANO model

        :type train_model: Theano.function
        :param train_model:

        :type validate_model: Theano.function
        :param validate_model:

        :type test_model: Theano.function
        :param test_model:

        :type n_train_batches: int
        :param n_train_batches: number of training batches

        :type n_valid_batches: int
        :param n_valid_batches: number of validation batches

        :type n_test_batches: int
        :param n_test_batches: number of testing batches

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type verbose: boolean
        :param verbose: to print out epoch summary or not to

        """
        
        
        print ('*************** Training ********************')
        # early-stopping parameters
        patience = 100000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.85  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter % 100 == 0) and verbose:
                    print('training @ iter = ', iter)
                cost_ij = train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    if verbose:
                        print('epoch %i, minibatch %i/%i, validation error %f %%' %
                            (epoch,
                             minibatch_index + 1,
                             n_train_batches,
                             this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            test_model(i)
                            for i in range(n_test_batches)
                        ]
                        test_score = numpy.mean(test_losses)

                        if verbose:
                            print(('epoch %i, minibatch %i/%i, test error of '
                                   'best model %f %%') %
                                  (epoch, minibatch_index + 1,
                                   n_train_batches,
                                   test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()

        # Retrieve the name of function who invokes train_nn() (caller's name)
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)

        # Print out summary
        print('Optimization complete.')
        print('Best validation error of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(('The training process for function ' +
               calframe[1][3] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

            

def construct_model(dataset='cifar-10-matlab.tar.gz',
                        learning_rate=0.01, 
                        nkerns=[32, 64], 
                        filter_size=[(3,3),(3,3)],
                        pool_size = [(2,2),(2,2)], 
                        hidden_layers = [4096,512],
                        hidden_activations=[T.tanh, T.tanh],
                        batch_size=512,
                        n_epochs=128):
        """ Demonstrates lenet on MNIST dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: path to the dataset used for training /testing (MNIST here)

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """ 
        
        rng = numpy.random.RandomState(23455)
        print ('loading started')
        ds_rate=None
        datasets = load_data(ds_rate=ds_rate,theano_shared=True)
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        #print('Current training data size is %i'%train_set_x.shape[0])
        #print('Current validation data size is %i'%valid_set_x.shape[0])
        #print('Current test data size is %i'%test_set_x.shape[0])

        # compute number of minibatches for training, validation and testing
        
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches //= batch_size
        n_valid_batches //= batch_size
        n_test_batches //= batch_size
        print ('loading complete')
        
        
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch

        # start-snippet-1
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')

        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        layer0_input = x.reshape((batch_size, 3, 32, 32))
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 3, 32, 32),
            filter_shape=(nkerns[0], 3, filter_size[0][0] , filter_size[0][1]),
            poolsize=pool_size[0]
        ) 

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        new_input_size = (32-filter_size[0][0]+1) / pool_size[0][0]

        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], 
                         new_input_size, 
                         new_input_size),
            filter_shape=(nkerns[1], nkerns[0], filter_size[1][0], filter_size[1][1]),
            poolsize=pool_size[1]
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        layer2_input = layer1.output.flatten(2)
        
        #print ('Layer2 output shape = ' + str(layer2_input.shape))
        
        new_input_size = (new_input_size - filter_size[1][0] + 1) / pool_size[1][0]

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * new_input_size * new_input_size,
            n_out=hidden_layers[0],
            activation=hidden_activations[0]
        )
        
        layer3 = HiddenLayer(
            rng,
            input=layer2.output,
            n_in=hidden_layers[0],
            n_out=hidden_layers[1],
            activation=hidden_activations[1]
        )
        
        # classify the values of the fully-connected sigmoidal layer
        layer4 = LogisticRegression(input=layer3.output, n_in=hidden_layers[1], n_out=10)

        # the cost we minimize during training is the NLL of the model
        cost = layer4.negative_log_likelihood(y)

        # create a function to compute the mistakes that are made by the model
        test_model = theano.function(
            [index],
            layer4.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        validate_model = theano.function(
            [index],
            layer4.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # create a list of all model parameters to be fit by gradient descent
        params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

        # create a list of gradients for all model parameters
        grads = T.grad(cost, params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

        train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
            
        )
        print ('model built')
        train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)
        # end-snippet-1


def test_lenet():
    construct_model()
    
        

#Problem 2.1
#Write a function to add translations
#Implement a convolutional neural network with the translation method for augmentation
def translate_image(img, x, y):
    img = numpy.array(numpy.reshape(img,(3,32,32))).transpose(1,2,0)
    img2 = numpy.zeros((32,32,3))
    for a in range(x,32):
        for b in range(y,32):
            img2[a][b] = img[a][b]
    
    img2 = img2.transpose(2,0,1).flatten()
    return img2

def construct_model_2(augmentation,
                        dataset='cifar-10-matlab.tar.gz',
                        learning_rate=0.01, 
                        nkerns=[32, 64], 
                        filter_size=[(3,3),(3,3)],
                        pool_size = [(2,2),(2,2)], 
                        hidden_layers = [4096,512],
                        hidden_activations=[T.tanh, T.tanh],
                        batch_size=512,
                        n_epochs=128):
        """ Demonstrates lenet on MNIST dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: path to the dataset used for training /testing (MNIST here)

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """
        
        rng = numpy.random.RandomState(23455)
        print ('loading started')
        ds_rate=None
        datasets = load_data_augmentation(augmentation,ds_rate=ds_rate,theano_shared=True)
        #datasets = load_data(ds_rate=ds_rate,theano_shared=True)
        
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        print('Current training data size is %i'%train_set_x.get_value(borrow=True).shape[0])
        print('Current validation data size is %i'%valid_set_x.get_value(borrow=True).shape[0])
        print('Current test data size is %i'%test_set_x.get_value(borrow=True).shape[0])

        # compute number of minibatches for training, validation and testing
        
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches //= batch_size
        n_valid_batches //= batch_size
        n_test_batches //= batch_size
        print ('loading complete')
        
        
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch

        # start-snippet-1
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')

        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        layer0_input = x.reshape((batch_size, 3, 32, 32))
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 3, 32, 32),
            filter_shape=(nkerns[0], 3, filter_size[0][0] , filter_size[0][1]),
            poolsize=pool_size[0]
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        new_input_size = (32-filter_size[0][0]+1) / pool_size[0][0]

        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], 
                         new_input_size, 
                         new_input_size),
            filter_shape=(nkerns[1], nkerns[0], filter_size[1][0], filter_size[1][1]),
            poolsize=pool_size[1]
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        layer2_input = layer1.output.flatten(2)
        
        #print ('Layer2 output shape = ' + str(layer2_input.shape))
        
        new_input_size = (new_input_size - filter_size[1][0] + 1) / pool_size[1][0]

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * new_input_size * new_input_size,
            n_out=hidden_layers[0],
            activation=hidden_activations[0]
        )
        
        layer3 = HiddenLayer(
            rng,
            input=layer2.output,
            n_in=hidden_layers[0],
            n_out=hidden_layers[1],
            activation=hidden_activations[1]
        )
        
        # classify the values of the fully-connected sigmoidal layer
        layer4 = LogisticRegression(input=layer3.output, n_in=hidden_layers[1], n_out=10)

        # the cost we minimize during training is the NLL of the model
        cost = layer4.negative_log_likelihood(y)

        # create a function to compute the mistakes that are made by the model
        test_model = theano.function(
            [index],
            layer4.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        validate_model = theano.function(
            [index],
            layer4.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # create a list of all model parameters to be fit by gradient descent
        params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

        # create a list of gradients for all model parameters
        grads = T.grad(cost, params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

        train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        print ('model built')
        train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)
        

def test_lenet_translation():
    construct_model_2(augmentation = ['Translation'])
    

#Problem 2.2
#Write a function to add roatations
def rotate_image(img, x):
    img = numpy.array(numpy.reshape(img,(3,32,32))).transpose(1,2,0)
    #img2 = numpy.zeros((32,32,3))
    img2 = scipy.ndimage.interpolation.rotate(img, x, axes=(0, 1, 0),reshape=False)
    
    img2 = img2.transpose(2,0,1).flatten()
    return img2
    
#Implement a convolutional neural network with the rotation method for augmentation
def test_lenet_rotation():
    construct_model_2(augmentation = ['Rotation'])


#Problem 2.3
#Write a function to flip images

def flip_image(img,x):
    img = numpy.array(numpy.reshape(img,(3,32,32))).transpose(1,2,0)
    if x:
        img2 = numpy.flipud(img)
    else:
        img2 = img
    img2 = img2.transpose(2,0,1).flatten()
    return img2

#Implement a convolutional neural network with the flip method for augmentation
def test_lenet_flip():
    construct_model_2(augmentation = ['Flip'])
    

#Problem 2.4
#Write a function to add noise, it should at least provide Gaussian-distributed and uniform-distributed noise with zero mean

def noise_injection(img, noise_typ):
    if noise_typ == "Gaussian":
        img = numpy.array(numpy.reshape(img,(3,32,32))).transpose(1,2,0)
        ch,row,col= img.shape
        mean = 0
        var = 0.005
        sigma = var**0.5
        gauss = numpy.random.normal(mean,sigma,(ch,row,col))
        gauss = gauss.reshape(ch,row,col)
        noisy = img + gauss
        img2 = noisy.transpose(2,0,1).flatten()
        return img2
    if noise_typ == "Uniform":
        img = numpy.array(numpy.reshape(img,(3,32,32))).transpose(1,2,0)
        ch,row,col= img.shape
        high = 0.01
        low = 0.01
        uni = numpy.random.uniform(low,high,(ch,row,col))
        uni = uni.reshape(ch,row,col)
        noisy = img + uni
        img2 = noisy.transpose(2,0,1).flatten()
        return img2
        

#Implement a convolutional neural network with the augmentation of injecting noise into input
def test_lenet_inject_noise_input():
    construct_model_2(augmentation = ['Noise'])
    
#Problem 3 
#Implement a convolutional neural network to achieve at least 80% testing accuracy on CIFAR-dataset 

def drop(input, p=0.5): 
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout is applied
    
    :type p: float or double between 0. and 1. 
    :param p: p probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.
    
    """            
    rng = numpy.random.RandomState(1234)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask

def momentum(cost, params, learning_rate, momentum):
    grads = theano.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        mparam_i = theano.shared(numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))
        v = momentum * mparam_i - learning_rate * g
        updates.append((mparam_i, v))
        updates.append((p, p + v))
    return updates


class DropoutHiddenLayer(object):
    def __init__(self, rng, is_train, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh, p=0.5):
        
        """
        Hidden unit activation is given by: activation(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        
        :type is_train: theano.iscalar   
        :param is_train: indicator pseudo-boolean (int) for switching between training and prediction

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
                           
        :type p: float or double
        :param p: probability of NOT dropping out a unit   
        """
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        
        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        
        output = activation(lin_output)
        
        # multiply output and drop -> in an approximation the scaling effects cancel out 
        train_output = drop(output,p)
        
        #is_train is a pseudo boolean theano variable for switching between training and prediction 
        self.output = T.switch(T.neq(is_train, 0), train_output, p*output)
        
        # parameters of the model
        self.params = [self.W, self.b]
        
class LeNetConvPoolDropoutLayer(object):
        """Pool Layer of a convolutional network """

        def __init__(self, rng, input, is_train, filter_shape, image_shape, p=0.8, poolsize=(2, 2)):
            """ 
            Allocate a LeNetConvPoolLayer with shared variable internal parameters.

            :type rng: numpy.random.RandomState
            :param rng: a random number generator used to initialize weights

            :type input: theano.tensor.dtensor4
            :param input: symbolic image tensor, of shape image_shape

            :type filter_shape: tuple or list of length 4
            :param filter_shape: (number of filters, num input feature maps,
                                  filter height, filter width)

            :type image_shape: tuple or list of length 4
            :param image_shape: (batch size, num input feature maps,
                                 image height, image width)

            :type poolsize: tuple or list of length 2
            :param poolsize: the downsampling (pooling) factor (#rows, #cols)
            """

            assert image_shape[1] == filter_shape[1]
            self.input = input

            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                       numpy.prod(poolsize))
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)

            # convolve input feature maps with filters
            conv_out = conv2d(
                input=input,
                filters=self.W,
                filter_shape=filter_shape,
                input_shape=image_shape,
                border_mode = 'half'
            )
            
            # pool each feature map individually, using maxpooling
            pooled_out = pool.pool_2d(
                input=conv_out,
                ds=poolsize,
                ignore_border=True
            )

            # add the bias term. Since the bias is a vector (1D array), we first
            # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
            # thus be broadcasted across mini-batches and feature map
            # width & height
            
            output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            train_output = drop(output,p)
        
            #is_train is a pseudo boolean theano variable for switching between training and prediction 
            self.output = T.switch(T.neq(is_train, 0), train_output, p*output)

            # store parameters of this layer
            self.params = [self.W, self.b]

            # keep track of model input
            self.input = input


def construct_model_3(dataset='cifar-10-matlab.tar.gz',
                        learning_rate=0.01, 
                        nkerns=[128, 128, 256], 
                        filter_size=[(3,3),(3,3),(3,3)],
                        pool_size = [(2,2),(2,2),(2,2)], 
                        #hidden_layers = [8192,2048,1024],
                        hidden_layers = [2048,2048],
                        hidden_activations=[T.nnet.relu, T.nnet.relu],
                        batch_size=256,
                        n_epochs=100):
        """ Demonstrates lenet on MNIST dataset

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: path to the dataset used for training /testing (MNIST here)

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer
        """
        
        rng = numpy.random.RandomState(23455)
        print ('loading started')
        ds_rate=None
        
        #datasets = load_data_augmentation(["Translation"],ds_rate=ds_rate,theano_shared=True)
        
        datasets = load_data(ds_rate=ds_rate,theano_shared=True)
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        #print('Current training data size is %i'%train_set_x.shape[0])
        #print('Current validation data size is %i'%valid_set_x.shape[0])
        #print('Current test data size is %i'%test_set_x.shape[0])

        # compute number of minibatches for training, validation and testing
        
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_train_batches //= batch_size
        n_valid_batches //= batch_size
        n_test_batches //= batch_size
        print ('loading complete')
        
        training_enabled = T.iscalar('training_enabled')
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch

        # start-snippet-1
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print('... building the model')

        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        layer0_input = x.reshape((batch_size, 3, 32, 32))
        
        '''layer00 = LeNetConvPoolDropoutLayer(
            rng,
            input=layer0_input,
            is_train=training_enabled,
            image_shape=(batch_size, 3, 32, 32),
            filter_shape=(nkerns[0], 3, filter_size[0][0] , filter_size[0][1]),
            poolsize=(1,1),
            mode = 'half',
            p = 1
        )
        
        layer01 = LeNetConvPoolDropoutLayer(
            rng,
            input=layer00.output,
            is_train=training_enabled,
            image_shape=(batch_size, nkerns[0], 32, 32),
            filter_shape=(nkerns[0], nkerns[0], filter_size[0][0] , filter_size[0][1]),
            poolsize=pool_size[0],
            p = 0.8
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (14-3+1, 14-3+1) = (12, 12)
        # maxpooling reduces this further to (12/2, 12/2) = (6, 6)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 6, 6)
        new_input_size = (32-filter_size[0][0]+1) / pool_size[0][0]

        layer10 = LeNetConvPoolDropoutLayer(
            rng,
            input=layer01.output,
            is_train=training_enabled,
            image_shape=(batch_size, nkerns[0], 
                         new_input_size, 
                         new_input_size),
            filter_shape=(nkerns[1], nkerns[0], filter_size[1][0], filter_size[1][1]),
            poolsize=(1,1),
            mode='half',
            p = 1
        )
        
        layer11 = LeNetConvPoolDropoutLayer(
            rng,
            input=layer10.output,
            is_train=training_enabled,
            image_shape=(batch_size, nkerns[1], 
                         new_input_size, 
                         new_input_size),
            filter_shape=(nkerns[1], nkerns[1], filter_size[1][0], filter_size[1][1]),
            poolsize=pool_size[1],
            p = 0.8
        )
        
        new_input_size = (new_input_size -filter_size[1][0]+1) / pool_size[1][0]
        
        layer20 = LeNetConvPoolDropoutLayer(
            rng,
            input=layer11.output,
            is_train=training_enabled,
            image_shape=(batch_size, nkerns[1], 
                         new_input_size, 
                         new_input_size),
            filter_shape=(nkerns[2], nkerns[1], filter_size[2][0], filter_size[2][1]),
            poolsize=(1,1),
            mode= 'half',
            p = 1
        )
        
        layer21 = LeNetConvPoolDropoutLayer(
            rng,
            input=layer20.output,
            is_train=training_enabled,
            image_shape=(batch_size, nkerns[2], 
                         new_input_size, 
                         new_input_size),
            filter_shape=(nkerns[2], nkerns[2], filter_size[2][0], filter_size[2][1]),
            poolsize=pool_size[2],
            p = 0.8
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        layer3_input = layer21.output.flatten(2)
        
        #print ('Layer2 output shape = ' + str(layer2_input.shape))
        
        new_input_size = (new_input_size - filter_size[2][0] + 1) / pool_size[2][0]'''
        
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (32-5+1 , 32-5+1) = (28, 28)
        # maxpooling reduces this further to (24/2, 24/2) = (14, 14)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 14, 14)
        layer0 = LeNetConvPoolDropoutLayer(
            rng,
            input=layer0_input,
            is_train=training_enabled,
            image_shape=(batch_size, 3, 32, 32),
            filter_shape=(nkerns[0], 3, filter_size[0][0] , filter_size[0][1]),
            poolsize=pool_size[0],
            p = 0.8
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (14-3+1, 14-3+1) = (12, 12)
        # maxpooling reduces this further to (12/2, 12/2) = (6, 6)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 6, 6)
        #new_input_size = (32-filter_size[0][0]+1) / pool_size[0][0]
        new_input_size = 16
        
        layer1 = LeNetConvPoolDropoutLayer(
            rng,
            input=layer0.output,
            is_train=training_enabled,
            image_shape=(batch_size, nkerns[0], 
                         new_input_size, 
                         new_input_size),
            filter_shape=(nkerns[1], nkerns[0], filter_size[1][0], filter_size[1][1]),
            poolsize=pool_size[1],
            p = 0.8
        )
        
        #new_input_size = (new_input_size -filter_size[1][0]+1) / pool_size[1][0]
        new_input_size = 8
        
        layer2 = LeNetConvPoolDropoutLayer(
            rng,
            input=layer1.output,
            is_train=training_enabled,
            image_shape=(batch_size, nkerns[1], 
                         new_input_size, 
                         new_input_size),
            filter_shape=(nkerns[2], nkerns[1], filter_size[2][0], filter_size[2][1]),
            poolsize=pool_size[2],
            p = 0.8
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        layer3_input = layer2.output.flatten(2)
        
        #print ('Layer2 output shape = ' + str(layer2_input.shape))
        
        #new_input_size = (new_input_size - filter_size[2][0] + 1) / pool_size[2][0]
        new_input_size = 4
        
        # construct a fully-connected sigmoidal layer
        layer3 = DropoutHiddenLayer(
            rng,
            input=layer3_input,
            is_train=training_enabled,
            n_in=nkerns[2] * new_input_size * new_input_size,
            n_out=hidden_layers[0],
            activation=hidden_activations[0],
            p = 0.8
        )
        
        layer4 = DropoutHiddenLayer(
            rng,
            input=layer3.output,
            is_train=training_enabled,
            n_in=hidden_layers[0],
            n_out=hidden_layers[1],
            activation=hidden_activations[1],
            p=0.8
        )
        
        # classify the values of the fully-connected sigmoidal layer
        layer6 = LogisticRegression(input=layer4.output, n_in=hidden_layers[1], n_out=10)

        # the cost we minimize during training is the NLL of the model
        cost = layer6.negative_log_likelihood(y)

        # create a function to compute the mistakes that are made by the model
        test_model = theano.function(
            [index],
            layer6.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size],
                training_enabled: numpy.cast['int32'](0)
            }
        )

        validate_model = theano.function(
            [index],
            layer6.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size],
                training_enabled: numpy.cast['int32'](0)
            }
        )

        # create a list of all model parameters to be fit by gradient descent
        params = layer6.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
        #layer10.params + layer20.params + layer00.params 

        # create a list of gradients for all model parameters
        #grads = T.grad(cost, params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        '''updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ] ''' 
        
        
        updates = momentum(cost, params, learning_rate, momentum=0.9)

        train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size],
                training_enabled: numpy.cast['int32'](1)
            }
        )
        print ('model built')
        train_nn(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = True)


def MY_lenet():
    construct_model_3()

'''
#Problem4
#Implement the convolutional neural network depicted in problem4 
def MY_CNN():
'''

