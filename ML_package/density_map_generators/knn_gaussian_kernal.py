import numpy as np
import scipy
import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt
import h5py
import PIL.Image as Image
from matplotlib import cm as CM
    #User's modules
from dm_generator import *

class KNN_Gaussian_Kernal_DMGenerator(DensityMapGenerator):
    
    

    def generate_densitymap(self,image,pointsList):
        '''
        This code use k-nearst, will take one minute or more to generate a density-map with one thousand people.

        points: a two-dimension list of pedestrians' annotation with the order [[col,row],[col,row],...].
        image_shape: the shape of the image, same as the shape of required density-map. (row,col). Note that can not have channel.

        return:
        density: the density-map we want. Same shape as input image but only has one channel.

        example:
        points: three pedestrians with annotation:[[163,53],[175,64],[189,74]].
        image_shape: (768,1024) 768 is row and 1024 is column.
        '''
        image_shape=[image.shape[0],image.shape[1]]
        print("Shape of current image: ",image_shape,". Totally need generate ",len(pointsList),"gaussian kernels.")
        density_map = np.zeros(image_shape, dtype=np.float32)
        ground_truth_count = len(pointsList)
        if ground_truth_count == 0:
            return density_map

        leafsize = 2048
        # build kdtree
        tree = scipy.spatial.KDTree(pointsList.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(pointsList, k=4)

        print ('generate density...')
        for i, pt in enumerate(pointsList):
            pt2d = np.zeros(image_shape, dtype=np.float32)
            if int(pt[1])<image_shape[0] and int(pt[0])<image_shape[1]:
                pt2d[int(pt[1]),int(pt[0])] = 1.
            else:
                continue
            if ground_truth_count > 1:
                sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            else:
                sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
            density_map += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        print ('done.')
        return density_map