"""
Source Code for Homework 3 of ECBM E4040, Fall 2016, Columbia University

This code contains implementation of several utility funtions for the homework.

Instructor: Prof.  Zoran Kostic

"""
import os
import sys
import numpy
import scipy.io
import tarfile
import theano
import random
import theano.tensor as T

def translate_image(img, x, y):
    img = numpy.array(numpy.reshape(img,(3,32,32))).transpose(1,2,0)
    img2 = numpy.zeros((32,32,3))
    for a in range(x,32):
        for b in range(y,32):
            img2[a][b] = img[a][b]
    
    img2 = img2.transpose(2,0,1).flatten()
    return img2

def rotate_image(img, x):
    img = numpy.array(numpy.reshape(img,(3,32,32))).transpose(1,2,0)
    #img2 = numpy.zeros((32,32,3))
    img2 = scipy.ndimage.interpolation.rotate(img, x, axes=(0, 1, 0),reshape=False)
    
    img2 = img2.transpose(2,0,1).flatten()
    return img2

def flip_image(img,x):
    img = numpy.array(numpy.reshape(img,(3,32,32))).transpose(1,2,0)
    if x:
        img2 = numpy.flipud(img)
    else:
        img2 = img
    img2 = img2.transpose(2,0,1).flatten()
    return img2

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

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def dataset_augmentation(augmentation, data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    new_x = []
    new_y = []
    if ('Translation' in augmentation):
        for x in data_x:
            new_x.append(translate_image(x,random.randint(1, 11),random.randint(1, 11)))
        new_y.extend(data_y)
    if ('Rotation' in augmentation):
        for x in data_x:
            new_x.append(rotate_image(x,random.randint(0, 360)))
        new_y.extend(data_y)
    if ('Flip' in augmentation):
        for x in data_x:
            new_x.append(flip_image(x,1))
        new_y.extend(data_y)
    if ('Noise' in augmentation):
        for x in data_x:
            new_x.append(noise_injection(x,"Gaussian"))
        new_y.extend(data_y)
            
    new_x = numpy.concatenate((data_x, new_x), axis=0)
    new_y = numpy.concatenate((data_y, new_y), axis=0)
    new_data_xy = new_x, new_y
    
    return new_data_xy

def load_data(ds_rate=None, theano_shared=True):
    ''' Loads the SVHN dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    if ds_rate is not None:
        assert(ds_rate > 1.)

    # Download the CIFAR-10 dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        #f_name = new_path.replace("src/../data/%s"%dataset, "data/") 
        f_name = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data"
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'https://www.cs.toronto.edu/~kriz/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path) 
             
        tar = tarfile.open(new_path)
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name,f_name)
        tar.close()              
        
        return f_name
    
    f_name=check_dataset('cifar-10-matlab.tar.gz')
    
    train_batches=os.path.join(f_name,'cifar-10-batches-mat/data_batch_1.mat')
    
    
    # Load data and convert data format
    train_batches=['data_batch_1.mat','data_batch_2.mat','data_batch_3.mat','data_batch_4.mat','data_batch_5.mat']
    train_batch=os.path.join(f_name,'cifar-10-batches-mat',train_batches[0])
    train_set=scipy.io.loadmat(train_batch)
    train_set['data']=train_set['data']/255.
    for i in range(4):
        train_batch=os.path.join(f_name,'cifar-10-batches-mat',train_batches[i+1])
        temp=scipy.io.loadmat(train_batch)
        train_set['data']=numpy.concatenate((train_set['data'],temp['data']/255.),axis=0)
        train_set['labels']=numpy.concatenate((train_set['labels'].flatten(),temp['labels'].flatten()),axis=0)
    
    test_batches=os.path.join(f_name,'cifar-10-batches-mat/test_batch.mat')
    test_set=scipy.io.loadmat(test_batches)
    test_set['data']=test_set['data']/255.
    test_set['labels']=test_set['labels'].flatten()
    
    train_set=(train_set['data'],train_set['labels'])
    test_set=(test_set['data'],test_set['labels'])
    

    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)
        train_set = [x[:train_set_len] for x in train_set]

    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//5):] for x in train_set]
    train_set = [x[:-(train_set_len//5)] for x in train_set]

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval

def load_data_augmentation(augmentation, ds_rate=None, theano_shared=True):
    ''' Loads the SVHN dataset

    :type ds_rate: float
    :param ds_rate: downsample rate; should be larger than 1, if provided.

    :type theano_shared: boolean
    :param theano_shared: If true, the function returns the dataset as Theano
    shared variables. Otherwise, the function returns raw data.
    '''
    if ds_rate is not None:
        assert(ds_rate > 1.)

    # Download the CIFAR-10 dataset if it is not present
    def check_dataset(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        #f_name = new_path.replace("src/../data/%s"%dataset, "data/") 
        f_name = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data"
        )
        if (not os.path.isfile(new_path)):
            from six.moves import urllib
            origin = (
                'https://www.cs.toronto.edu/~kriz/' + dataset
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, new_path) 
             
        tar = tarfile.open(new_path)
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name,f_name)
        tar.close()              
        
        return f_name
    
    f_name=check_dataset('cifar-10-matlab.tar.gz')
    
    train_batches=os.path.join(f_name,'cifar-10-batches-mat/data_batch_1.mat')
    
    
    # Load data and convert data format
    train_batches=['data_batch_1.mat','data_batch_2.mat','data_batch_3.mat','data_batch_4.mat','data_batch_5.mat']
    train_batch=os.path.join(f_name,'cifar-10-batches-mat',train_batches[0])
    train_set=scipy.io.loadmat(train_batch)
    train_set['data']=train_set['data']/255.
    for i in range(4):
        train_batch=os.path.join(f_name,'cifar-10-batches-mat',train_batches[i+1])
        temp=scipy.io.loadmat(train_batch)
        train_set['data']=numpy.concatenate((train_set['data'],temp['data']/255.),axis=0)
        train_set['labels']=numpy.concatenate((train_set['labels'].flatten(),temp['labels'].flatten()),axis=0)
    
    test_batches=os.path.join(f_name,'cifar-10-batches-mat/test_batch.mat')
    test_set=scipy.io.loadmat(test_batches)
    test_set['data']=test_set['data']/255.
    test_set['labels']=test_set['labels'].flatten()
    
    train_set=(train_set['data'],train_set['labels'])
    test_set=(test_set['data'],test_set['labels'])
    

    # Downsample the training dataset if specified
    train_set_len = len(train_set[1])
    if ds_rate is not None:
        train_set_len = int(train_set_len // ds_rate)
        train_set = [x[:train_set_len] for x in train_set]

    # Extract validation dataset from train dataset
    valid_set = [x[-(train_set_len//5):] for x in train_set]
    train_set = [x[:-(train_set_len//5)] for x in train_set]

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.       
    
    train_set = dataset_augmentation(augmentation,train_set)
    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)
        
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval
