"""
Source Code for Homework 4.b of ECBM E4040, Fall 2016, Columbia University

"""
from __future__ import print_function

import os
import timeit
import inspect
import sys
import numpy
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from hw4_utils import contextwin, shared_dataset, load_data, shuffle, conlleval, check_dir
from hw4_nn import myMLP, train_nn


def train_nn2(train_model, validate_model, test_model,
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

    # early-stopping parameters
    patience = 1000000  # look as this many examples regardless
    patience_increase = 10  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.9995  # a relative improvement of this much is
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
        if (epoch % 10000 == 0):
            print (epoch)
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
                        print(('     epoch %i, minibatch %i/%i, test error of '
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
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    

def gen_parity_pair(nbit, num):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """
    X = numpy.random.randint(2, size=(num, nbit))
    Y = numpy.mod(numpy.sum(X, axis=1), 2)
    
    return X, Y

#TODO: implement RNN class to learn parity function
class RNN(object) :
    def __init__(self, input_dim, h_dim, output_dim, normal=True):
        """
        """
        nh = h_dim
        
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (input_dim, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, output_dim))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(output_dim,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))

        # bundle
        self.params = [self.wx, self.wh, self.w,
                       self.bh, self.b, self.h0]

        # as many columns as context window size
        # as many lines as words in the sentence
        idxs = T.imatrix()
        #x = self.emb[idxs].reshape((idxs.shape[0], input_dim))
        x = idxs.reshape((1000, input_dim))
        y_sentence = T.ivector('y_sentence')  # labels


        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])

        #p_y_given_x_sentence = s[:, 0, :]
        p_y_given_x_sentence = s
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)
        y_pred = y_pred.reshape(1000,1)
        self.y_pred = y_pred
        
        #prediction=T.argmax(o,axis=1)
        o_error=T.sum(T.nnet.categorical_crossentropy(y_pred,y_sentence))

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        
        

        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates,
                                              allow_input_downcast=True)
        self.ce_error = theano.function([x, y_sentence], o_error, allow_input_downcast = True)
        

    def train(self, x, y, learning_rate):

        words = x.reshape(1000, 8)
        labels = y
        #print (x.shape, y.shape)
        self.sentence_train(words, labels, learning_rate)


    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))
    
    def calculate_total_loss(self,X,Y):
        print (X.shape, Y.shape)
        return numpy.sum([self.ce_error(x,y) for x,y in zip(X,Y)])

    def calculate_loss(self,X,Y):
        # Divide calculate_loss by the number of words
        num_words=len(Y)
        #num_words = numpy.sum([len(y) for y in Y])
        #num_words = Y.shape[0]
        return self.calculate_total_loss(X,Y)/float(num_words)
    
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        

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
        print (self.y_pred.shape)
        print (len(y))
        count = 0
        for a in range(len(y)):
            if self.y_pred[a] != y[a]:
                count += 1
        return count*100 / len(y)"""
        #self.y_pred.eval()
        return T.mean(T.neq(self.y_pred, y))
    
   
    def train_with_sgd(model,X_train,y_train,learning_rate=0.005,nepoch=1,evaluate_loss_after=5):
        # We keep track of lossed so we can plot them later
        losses=[]
        num_examples_seen=0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if(epoch % evaluate_loss_after==0):
                #loss=model.calculate_loss(X_train,y_train)
                loss=model.errors(y_train)
                losses.append((num_examples_seen,loss))
                time=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                print ("%s: Loss after num_examples_seen=%d epoch=%d:" %(time,num_examples_seen,epoch))
                # Adjust the learning rate if loss increases
                if (len(losses)>1 and losses[-1][1]>losses[-2][1]):
                    learning_rate=learning_rate*0.5
                    print ("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()
                # Saving model parameters
                #save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" %(model.hidden_dim,model.word_dim,time),model)
            # For each training examples...
            for i in range(len(y_train)):
                # One SGD Step
                #model.sgd_step(X_train[i],y_train[i],learning_rate)
                model.train(X_train[i],y_train[i],learning_rate)
                num_examples_seen+=1 
    

#TODO: implement LSTM class to learn parity function
class LSTM(object):
    pass



def test_mlp_parity(n_bit,n_hidden,n_hiddenLayers, epoch, learning_rate, batch_size, L1, L2):
    
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    
    batch_size=batch_size
    n_epochs=epoch
    L1_reg=L1
    L2_reg=L2
    learning_rate=learning_rate
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    '''classifier = MLP(
         rng=rng,
         input=x,
         n_in=32*32*3,
         n_hidden=840,
         n_out=10,
    )'''
    
    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=n_bit,
        n_hidden = n_hidden,
        n_hiddenLayers = n_hiddenLayers,
        n_out=2
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    
    #updates = adadelta(gparams, classifier.params)

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    train_nn2(train_model, validate_model, test_model,
            n_train_batches, n_valid_batches, n_test_batches, n_epochs,
            verbose = False)

    
def get_dataset(data_xy):
    data_x, data_y = data_xy
    return data_x, data_y
    
#TODO: build and train a RNN to learn parity function
def test_rnn_parity(n_bit):
    
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = get_dataset(train_set)
    valid_set_x, valid_set_y = get_dataset(valid_set)
    test_set_x, test_set_y = get_dataset(test_set)

    train_set_y = train_set_y.reshape(1000,1)
    #train_set_x, train_set_y = shared_dataset(train_set)
    #valid_set_x, valid_set_y = shared_dataset(valid_set)
    #test_set_x, test_set_y = shared_dataset(test_set)
    
    
    batch_size=20
    n_epochs=10
    L1_reg=0.00
    L2_reg=0.0001
    learning_rate=0.01
    
    '''n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size
    
    rnn = RNN(n_bit, 20, 1)

    lr = 0.01
	e = 1
	vals = []
	for i in xrange(train_set):
		c = rnn.train_step(train_set_x[i], train_set_y[i], lr)
		print "iteration {0}: {1}".format(i, np.sqrt(c))
		e = 0.1*np.sqrt(c) + 0.9*e
		if i % 100 == 0:
            count = 0
			for val in xrange(valid_set): '''
                
    
    n_train_batches = len(train_set_x) // batch_size
    n_valid_batches = len(valid_set_x) // batch_size
    n_test_batches = len(test_set_x) // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)
    
    model=RNN(input_dim=8, h_dim=100, output_dim=2)

    train_with_sgd(model,train_set_x,train_set_y,nepoch=n_epochs,learning_rate=learning_rate)

    
    
#TODO: build and train a LSTM to learn parity function
def test_lstm_parity(n_bit):
    
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    

    
if __name__ == '__main__':
    test_mlp_parity()
