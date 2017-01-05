import os
from os import walk
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.linalg as linalg

from PIL import Image

import theano
import theano.tensor as T


'''
Implement the functions that were not implemented and complete the
parts of main according to the instructions in comments.
'''

def reconstructed_image(D,c,num_coeffs,X_mean,im_num):
    '''
    This function reconstructs an image given the number of
    coefficients for each image specified by num_coeffs
    '''
    
    '''
        Parameters
    ---------------
    c: np.ndarray
        a n x m matrix  representing the coefficients of all the image blocks.
        n represents the maximum dimension of the PCA space.
        m is (number of images x n_blocks**2)

    D: np.ndarray
        an N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)

    im_num: Integer
        index of the image to visualize

    X_mean: np.ndarray
        a matrix representing the mean block.

    num_coeffs: Integer
        an integer that specifies the number of top components to be
        considered while reconstructing
    '''
    
    c_im = c[:num_coeffs,im_num]
    D_im = D[:,:num_coeffs]
    
    #TODO: Enter code below for reconstructing the image
    #......................
    #......................
    #X_recon_img = ........
    return X_recon_img

def plot_reconstructions(D,c,num_coeff_array,X_mean,im_num):
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number_of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,im_num))
            
    f.savefig('output/hw1b_{0}.png'.format(im_num))
    plt.close(f)
    
    
    
def plot_top_16(D, sz, imname):
    '''
    Plots the top 16 components from the basis matrix D.
    Each basis vector represents an image block of shape (sz, sz)

    Parameters
    -------------
    D: np.ndarray
        N x n matrix representing the basis vectors of the PCA space
        N is the dimension of the original space (number of pixels in a block)
        n represents the maximum dimension of the PCA space (assumed to be atleast 16)

    sz: Integer
        The height and width of each block

    imname: string
        name of file where image will be saved.
    '''
    #TODO: Obtain top 16 components of D and plot them
    
    raise NotImplementedError

    
def main():
    '''
    Read here all images(grayscale) from Fei_256 folder and collapse 
    each image to get an numpy array Ims with size (no_images, height*width).
    Make sure the images are read after sorting the filenames
    '''
    #TODO: Write a code snippet that performs as indicated in the above comment
    
    print 'starting 1'
    begFileName = "Fei_256/image"
    endFileName = ".jpg"
    numberOfImages = 10
    imageLength, imageBreadth = 256, 256
    Ims = np.zeros((numberOfImages,imageLength*imageBreadth),np.float32)
    for c in range(0,numberOfImages) :
        im = Image.open(begFileName + str(c) + endFileName)
        im= im.convert('L') #convert greyscale
        data = np.asarray( im )
        Ims[c] = data.flatten()
    
    #Ims = Ims.astype(np.float32)
    print '1'
    X_mn = np.mean(Ims, 0)
    X = Ims - np.repeat(X_mn.reshape(1, -1), Ims.shape[0], 0)
    print 'X shape '
    print X.shape
    
    '''
    #initial_A = np.dot(X.T, X)
    initial_d = np.random.randn(256*256)
    eig_val, eig_vec = np.linalg.svd(np.dot(X.T, X), full_matrices=False)
    
    #A = theano.shared(initial_A)
    d = theano.shared(initial_d)
    y = theano.shared(y)
    f = - T.dot(T.dot(d.T, A),d)
    
    print 'A shape + d shape'
    print A.shape

    def gradient_descent(A, lmd, d, f):
        A = np.dot(X.T, X) - lmd *np.dot(d,d.T)
        t = 1
        step_size = 0.2
        grad_f = T.grad(f, d)
        lmd0 = 1000  #a very large number
        while t < 200 and abs(lmd-lmd0) > 0.01:
            y = d - step_size * grad_f
            d = y/norm(y)
            t+=1
        lmd0 = lmd*1
        lmd = np.dot(np.dot(d.T, A),d)
        return (A, lmd, d)

    results, updates = theano.scan(gradient_descent, outputs_info=[A, lmd,d], n_steps=16)
    func = theano.function([A, lmd, d, f], results, updates= updates)

    p = func(A,lmd,d,f)
    print "p"
    p.shape'''

    '''
    Use theano to perform gradient descent to get top 16 PCA components of X
    Put them into a matrix D with decreasing order of eigenvalues

    If you are not using the provided AMI and get an error "Cannot construct a ufunc with more than 32 operands" :
    You need to perform a patch to theano from this pull(https://github.com/Theano/Theano/pull/3532)
    Alternatively you can downgrade numpy to 1.9.3, scipy to 0.15.1, matplotlib to 1.4.2
    '''
    
    #TODO: Write a code snippet that performs as indicated in the above comment
    init_d = (np.random.randn(65536,1) * 0.01)
    theano.shared(init_d)

    evals, evecs = linalg.eigh(np.dot(X, X.T))
    
    #idx = evals.argsort()[::-1]
    #print evecs.shape
    #evals = evals[idx]
    #evecs = evecs[:,idx]
    sort_perm = evals.argsort()

    evals.sort()     # <-- This sorts the list in place.
    evecs = evecs[:, sort_perm]
    evecs2 = (np.dot(X.T, evecs).T) 
    #evecs2 = evecs2.T
    #evals = eigenvalues 
    print '*********'
    print evals
    print evecs2.shape
    print evecs2[0].shape
    evecs=evecs2

    d = theano.shared(init_d, name="d")
    #d=init_d
    Xd = T.dot(X, d)
    print T.dot(evecs[0], d).shape
    print '^'
    cost = T.dot(Xd.T, Xd) - np.sum(evals[j]*T.dot(evecs[j], d)*T.dot(evecs[j], d) for j in xrange(numberOfImages))
    print cost.shape
    print cost[0]
    
    gd = T.grad(cost, d)
    y = d + learning_rate*gd
    update_expression = y / y.norm(2)
    
    print 2
    
    train = theano.function(
      inputs=[X],
      outputs=[y_outputs],
        updates=((v, update_expression),)
    )
    
    for i in xrange(16):
        cost = train(X)
        
    print 3   
    for i in range(0, 200, 10):
        plot_reconstructions(D=D, c=c, num_coeff_array=[1, 2, 4, 6, 8, 10, 12, 14, 16], X_mean=X_mean.reshape((256, 256)), im_num=i)

    plot_top_16(D, 256, 'output/hw1b_top16_256.png')


if __name__ == '__main__':
    main()
    
    