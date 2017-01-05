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

def reconstructed_image(D,c,num_coeffs,X_mean,n_blocks,im_num,sz):
    '''
    This function reconstructs an image X_recon_img given the number of
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
      

    n_blocks: Integer
        number of blocks comprising the image in each direction.
        For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4
    '''
    
    c_im = c[:num_coeffs,n_blocks*n_blocks*im_num:n_blocks*n_blocks*(im_num+1)]
    D_im = D[:,:num_coeffs] 
    
    #print c_im.shape
    #print D_im.shape
    
    R = np.dot(D_im,c_im) #+ X_mean
    
    X_recon_img = R.T
    #print 'X_recon shape = '
    #print X_recon_img.shape
    #print ''
    #print X_mean.shape
    
    X_dim = X_recon_img.size/X_recon_img[0].size
    for i in range(X_dim):
        X_recon_img[i] += X_mean.flatten()
        
    X_recon_img_2 = np.zeros((256,256))
    
    '''for i in range(256):
        for j in range(256):
            xi_i = (int) (i/sz)
            yi_i = (int) (j/sz)
            xj = i - xi_i*sz
            yj = j - yi_i*sz
            X_recon_img_2[i][j] = X_recon_img[xi_i*n_blocks+yi_i][sz*xj+yj]
    '''
    #print '******************'
    for i in range(X_dim):
        p = X_recon_img[i]
        p = p.reshape((sz,sz),)
        p = p.T
        xi_i = (int) (i/n_blocks)
        yi_i = i - xi_i*n_blocks
        #print i
        #print xi_i
        #print yi_i
        #print ''
        for x in range(sz):
            for y in range(sz):
                X_recon_img_2[(xi_i*sz)+x][(yi_i*sz)+y] = p[x][y] 
            
    X_recon_img_2 = X_recon_img_2.T
    '''print X_recon_img.shape
    print X_recon_img.size/X_recon_img[0].size
    
    X_dim = X_recon_img.size/X_recon_img[0].size
    B_dim = (256*256)/(n_blocks*n_blocks)
    
    X_recon_img=X_recon_img.reshape((X_dim,B_dim),)
    print X_recon_img.shape'''
    
    '''
    # Defining variables
    images = T.tensor4('images')
    neibs = T.nnet.neighbours.images2neibs(images, neib_shape=(256, 256))

    # Constructing theano function
    window_function = theano.function([images], neibs, allow_input_downcast=True)

    # Input tensor (one image 10x10)
    im_val = X_recon_img

    # Function application
    neibs_val = window_function(im_val)
    print neibs_val.shape
    print '******* \n'
    '''
    
    '''im_new = T.nnet.neighbours.neibs2images(neibs, (8, 8), (256,256))
    # Theano function definition
    inv_window = theano.function([neibs], im_new)
    # Function application
    im_new_val = inv_window(X_recon_img)'''
    #X_recon_img=X_recon_img.reshape((256,256),)
    #TODO: Enter code below for reconstructing the image X_recon_img
    #......................
    #......................
    #X_recon_img = ........
    return X_recon_img_2

def plot_reconstructions(D,c,num_coeff_array,X_mean,n_blocks,im_num,sz):
    
    '''
    Plots 9 reconstructions of a particular image using D as the basis matrix and coeffiecient
    vectors from c

    Parameters
    ------------------------
        num_coeff_array: Iterable
            an iterable with 9 elements representing the number of coefficients
            to use for reconstruction for each of the 9 plots
        
        c: np.ndarray
            a l x m matrix  representing the coefficients of all blocks in a particular image
            l represents the dimension of the PCA space used for reconstruction
            m represents the number of blocks in an image

        D: np.ndarray
            an N x l matrix representing l basis vectors of the PCA space
            N is the dimension of the original space (number of pixels in a block)

        n_blocks: Integer
            number of blocks comprising the image in each direction.
            For example, for a 256x256 image divided into 64x64 blocks, n_blocks will be 4

        X_mean: basis vectors represent the divergence from the mean so this
            matrix should be added to all reconstructed blocks

        im_num: Integer
            index of the image to visualize
    '''
    f, axarr = plt.subplots(3,3)
    for i in range(3):
        for j in range(3):
            plt.axes(axarr[i,j])
            plt.imshow(reconstructed_image(D,c,num_coeff_array[i*3+j],X_mean,n_blocks,im_num,sz),cmap='Greys_r')
            
    f.savefig('output/hw1a_{0}_im{1}.png'.format(n_blocks, im_num))
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
    #print 'p_16'

    d = D[:16]
    #print d.shape
    g, axarr2 = plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            plt.axes(axarr2[i,j])
            plt.imshow(d[i*4+j].reshape((sz,sz),),cmap='Greys_r')
    
    g.savefig('output/hw1a_top16_{0}.png'.format(sz))
    plt.close(g)    
    
    print 'complete'

    
def main():
    
    '''
    Read here all images(grayscale) from Fei_256 folder
    into an numpy array Ims with size (no_images, height, width).
    Make sure the images are read after sorting the filenames
    '''
    #TODO: Read all images into a numpy array of size (no_images, height, width)
    
    print 'starting 1'
    begFileName = "Fei_256/image"
    endFileName = ".jpg"
    numberOfImages = 200
    imageLength, imageBreadth = 256, 256
    npImages = np.zeros((numberOfImages,imageLength,imageBreadth))
    i_sort = []
    s_sort = []
    for c in range(0,numberOfImages) :
        i_sort.append(str(c))
    i_sort.sort()
    for c in i_sort:
        s_sort.append(c)
    #print s_sort
    count=0
    for c in s_sort :
        im = Image.open(begFileName + c + endFileName)
        im= im.convert('L') #convert greyscale
        data = np.asarray( im )
        npImages[count] = data
        count+=1
    #print npImages
    #print ''
    
    
    szs = [8, 32, 64]
    num_coeffs = [range(1, 10, 1), range(3, 30, 3), range(5, 50, 5)]
    #szs = [32]
    #num_coeffs = [range(3, 30, 3)]
    
    for sz, nc in zip(szs, num_coeffs):
        
        '''
        Divide here each image into non-overlapping blocks of shape (sz, sz).
        Flatten each block and arrange all the blocks in a
        (no_images*n_blocks_in_image) x (sz*sz) matrix called X
        '''
        
        #TODO: Write a code snippet that performs as indicated in the above comment
        
        
        n_blocks_in_image = (imageLength * imageBreadth) / (sz * sz)
        X = np.zeros((numberOfImages*n_blocks_in_image, sz * sz ))
        X_iter =0
        
        for c in range (0,numberOfImages) :
            im = Image.fromarray(npImages[c])
            block_iter = 0
            for i in range(0,imageLength,sz):
                for j in range(0,imageBreadth,sz):
                    box = (i, j, i+sz, j+sz)
                    a = im.crop(box)
                    data = np.asarray( a )
                    X[(c)*n_blocks_in_image + block_iter] = data.flatten()
                    block_iter +=1
                    
        #print X
        #print ''
        #print X.shape
        #print ''
        
    
        X_mean = np.mean(X, 0)
        X = X - np.repeat(X_mean.reshape(1, -1), X.shape[0], 0)
    
        '''
        Perform eigendecomposition on X^T X and arrange the eigenvectors
        in decreasing order of eigenvalues into a matrix D
        '''
        
        #TODO: Write a code snippet that performs as indicated in the above comment
        
        
        xtx = np.dot(X.transpose(), X)
        eig_val, eig_vec = np.linalg.eigh(xtx)
        
        idx = eig_val.argsort()[::-1]
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        eig_vecs_sorted = eig_vec
        eig_vals_sorted = eig_val
        #eig_vals_sorted = np.sort(eig_val)
        #eig_vecs_sorted = eig_vec[:, eig_val.argsort()]        

        '''print ''
        print 'eig_vecs'
        print eig_vecs_sorted.shape
        print ''
        print 'eig_vals'
        print eig_vals_sorted.shape
        print '''

        D = eig_vecs_sorted
        
        c = np.dot(D.T, X.T)
        
        #print 'D dims ' + str(D.shape)
        #print 'c_dims ' + str(c.shape)
        print sz, nc
        for i in range(0, numberOfImages, 10):
            print "Here " + str(i)
            plot_reconstructions(D=D, c=c, num_coeff_array=nc, X_mean=X_mean.reshape((sz, sz)), n_blocks=int(256/sz), im_num=i,sz=sz)

        plot_top_16(D, sz, imname='output/hw1a_top16_{0}.png'.format(sz))
        


if __name__ == '__main__':
    main()

    
