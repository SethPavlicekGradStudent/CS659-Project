## Overview
#The purpose of this script is to make a customized version of the grid search hough style voting for tuning hyperparameters 
#in the get features function for generating the description vectors for the images
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # no UI backend
import matplotlib.pyplot as plt
from utils import *
import math
import os

class IncomaptibleWindowSizes(Exception):
    pass

#################################################### SCRIPT KEYPOINT

def ANMS (x , y, r, maximum):
    #x is an array of length N
    #y is an array of length N
    #r is the cornerness score
    #max is the number of corners that are required
    i = 0
    j = 0
    NewList = []

    while i < len(x):

        minimum = 1000000000000 #random large value
        FirstCoordinate, SecondCoordinate = x[i], y[i]

        while j < len(x):

            CompareCoordinate1, CompareCoordinate2 = x[j], y[j]
            if (FirstCoordinate != CompareCoordinate1 and SecondCoordinate != CompareCoordinate2) and r[i] < r[j]:

                distance = math.sqrt((CompareCoordinate1 - FirstCoordinate)**2 + (CompareCoordinate2 - SecondCoordinate)**2)
                if distance < minimum:
                    minimum = distance

            j = j + 1
        NewList.append([FirstCoordinate, SecondCoordinate, minimum])

        i = i + 1
        j = 0

    NewList.sort(key = lambda t: t[2])
    NewList = NewList[len(NewList)-maximum:len(NewList)]

    return NewList


def get_interest_points(image, feature_width):

    alpha = 0.04
    threshold = 10000 # minimal value of Harris Score. Any points scored less than this threshold should be removed. 
    XCorners = [] # X-coordinate
    YCorners = [] # Y-coordinate
    RValues = []  # Cornerness value

    #Compute the size of the image.
    ImageRows = image.shape[0]
    ImageColumns = image.shape[1]

    #Use the soble filter to calculate the x and y derivative of the image. You might use the cv2.Sobel() as follows.
    Xderivative = cv2.Sobel(image, cv2.CV_64F,1,0,ksize=5)
    Yderivative = cv2.Sobel(image, cv2.CV_64F,0,1,ksize=5)

    #Define matrices Ixx, Iyy and Ixy
    Ixx = (Xderivative)*(Xderivative)
    Iyy = (Yderivative)*(Yderivative)
    Ixy = (Xderivative)*(Yderivative)

    filter1 = cv2.getGaussianKernel(ksize=4, sigma=2)
    Ixx = cv2.filter2D(Ixx, -1, filter1)
    Iyy = cv2.filter2D(Iyy, -1, filter1)
    Ixy = cv2.filter2D(Ixy, -1, filter1)

    #loop over the image to compute cornerness score of each pixel
    for i in range(ImageRows):
        for j in range(ImageColumns):
            score = Ixx[i][j]*Iyy[i][j] - (Ixy[i][j]**2) - (alpha * (Ixx[i][j]+Iyy[i][j])**2)
            if score >= threshold:
                XCorners.append(j)
                YCorners.append(i)
                RValues.append(score)

    XCorners = np.asarray(XCorners)
    YCorners = np.asarray(YCorners)
    RValues = np.asarray(RValues)

    #Use ANMS to evenly distribute the corners in the image.
    NewCorners = ANMS(XCorners, YCorners, RValues, 3025)
    NewCorners = np.asarray(NewCorners)

    #Return the x-y coordinates and cornerness score of the eligible corners.
    x = NewCorners[:,0]
    y = NewCorners[:,1]
    scales = NewCorners[:,2]

    return x,y, scales

#################################################### END SCRIPT KEYPOINT


def get_features(image, x, y, feature_width, mainWindowSize, smallerWindowSize, bins):
  #calculations based on hyperparamaters
  halfmainWindowSize = mainWindowSize // 2
  numRegions = (mainWindowSize // smallerWindowSize) ** 2
  numRegionsAcross = int(math.sqrt(numRegions))

    #the goal of this function is to extract a feature vector for each interest point
  #Round off the x and y coordinates to integers.
  x = np.rint(x)
  x = x.astype(int)
  y = np.rint(y)
  y = y.astype(int)

  #Define a gaussian filter.
  cutoff_frequency = 10
  filter1 = cv2.getGaussianKernel(ksize=4,sigma=cutoff_frequency)
  filter1 = np.dot(filter1, filter1.T)

  #Apply the gaussian filter to the image.
  image = cv2.filter2D(image, -1, filter1)
  ImageRows = image.shape[0]
  ImageColumns = image.shape[1]

  Xcoordinates = len(x)
  Ycoordinates = len(y)

  #pad image to prevent out of bounds error can play with how we pad it 
  padSize = halfmainWindowSize
  image = np.pad(image, padSize, mode='edge')

  if (mainWindowSize % smallerWindowSize != 0):
    # Raise the exception
    raise IncomaptibleWindowSizes("Incompatible large and small window sizes ")

  dim = numRegions * bins;  # feature dimension
  FeatureVectorIn = np.ones((Xcoordinates,dim)) # each row represents a vector of dim. 
  NormalizedFeature = np.zeros((Xcoordinates,dim))

  #loop over the corners generated by Harris
  for i in range(Xcoordinates):

    #Extract a 16X16 window centered at the corner pixel
    temp1 = int(x[i]) + padSize
    temp2 = int(y[i]) + padSize
    Window = image[temp2 -  halfmainWindowSize:temp2 + halfmainWindowSize, temp1 - halfmainWindowSize:temp1 + halfmainWindowSize]
    
    # write your own code to extract the feature vectors FeatureVectorIn for this 16 by 16 window. 
    gradient_base_orientations = np.gradient(Window)
    gradient_magnitudes = np.sqrt(gradient_base_orientations[0]**2 + gradient_base_orientations[1]**2)
    gradient_angles = np.rad2deg(np.mod(np.arctan2(gradient_base_orientations[0], gradient_base_orientations[1]), 2*np.pi)).astype('uint32')
    complete_feature_vector = np.zeros((numRegionsAcross, numRegionsAcross, bins))
    
    angle_floor = 360 // bins
    for j in range(Window.shape[0]):
      for k in range(Window.shape[0]):
        vector_y = j // (mainWindowSize // numRegionsAcross)
        vector_x = k // (mainWindowSize // numRegionsAcross)
        or_index = ((gradient_angles[j][k] - 1) // angle_floor) % bins
        
        if(or_index < (bins - 1)):
          weight_r = (gradient_angles[j][k] % angle_floor) / angle_floor
          weight_l = (angle_floor - (gradient_angles[j][k] % angle_floor)) / angle_floor
          complete_feature_vector[vector_y, vector_x, or_index] += weight_l * gradient_magnitudes[j][k]
          complete_feature_vector[vector_y,vector_x, or_index+1] += weight_r * gradient_magnitudes[j][k]
        else:
          complete_feature_vector[vector_y,vector_x, or_index - 1] += gradient_magnitudes[j][k]

    FeatureVectorIn[i] = complete_feature_vector.flatten()

    #Write your code to normalize the generated feature vector
    NormalizedFeature[i] = FeatureVectorIn[i]/np.linalg.norm(FeatureVectorIn[i],1)
    NormalizedFeature[i] = np.clip(FeatureVectorIn[i],0,0.2)
    NormalizedFeature[i] = NormalizedFeature[i]/np.linalg.norm(NormalizedFeature[i],1)
 
  #Return normalized feature vector
  fv = NormalizedFeature
  return fv

def compute_euclidean_distance(hist1, hist2):
    #histogram Eucleidan distance calcualtion
    squared_diff = (hist1 - hist2) ** 2
    sum_squared_diff = np.sum(squared_diff)
    euclidean_distance = np.sqrt(sum_squared_diff)
    
    return euclidean_distance


def match_features(features1, features2, x1, y1, x2, y2, mainWindowSize, bins, numRegions):

    Distance = np.zeros((features1.shape[0], features2.shape[0])) # Euclidean distance. 
    Value = [] # distance of matched feature
    Hitx = [] # x-coordinate of matched feature
    Hity = [] # y-coordinate of matched feature

    #loop over the features1, write your code to find the matched feature in features 2 that has minimal disance. 
    for i in range(features1.shape[0]):
        for j in range(features2.shape[0]):
            sum_equils = 0
            for k in range(mainWindowSize):
                hist_index = k * bins
                sum_equils += compute_euclidean_distance(features1[i,hist_index:hist_index + bins],features2[j,hist_index:hist_index + bins])
            Distance[i][j] = sum_equils / numRegions

    
    for i in range(features1.shape[0]):
        index = np.argmin(Distance[i])
        Value.append(np.min(Distance[i]))
        Hitx.append(i)
        Hity.append(index)


    # convert to Numpy Array. 
    Xposition = np.asarray(Hitx).astype('int32')
    Yposition = np.asarray(Hity).astype('int32')
    matches = np.stack((Xposition,Yposition), axis = -1)
    confidences = np.asarray(Value)

    sorted_indices = np.argsort(confidences)
    matches = matches[sorted_indices]
    confidences = confidences[sorted_indices]

    return matches, confidences

#check if the output image file exists for we dont rerun si ulations we have already done
def file_exists(file_path):
    return os.path.exists(file_path)

def main():
    #need to set working directory as the current directory for paths to work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    image1 = load_image('data/Notre Dame/A.jpg')
    image2 = load_image('data/Notre Dame/B.jpg')
    eval_file = 'data/Notre Dame/A_to_B.pkl'

    scale_factor = 0.5
    image1 = cv2.resize(image1, (0, 0), fx=scale_factor, fy=scale_factor)
    image2 = cv2.resize(image2, (0, 0), fx=scale_factor, fy=scale_factor)
    image1_bw = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    image2_bw = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    feature_width = 16 # width and height of each local feature, in pixels. 
    ########## interest point detection ##########
    print('get_interest_points()...image 1')
    x1, y1, scales1= get_interest_points(image1_bw, feature_width)
    print('get_interest_points()... image 2')
    x2, y2, scales2 = get_interest_points(image2_bw, feature_width)

    # no visualizing the 
    print('visualizing the interest points')
    # Visualize the interest points
    c1 = show_interest_points(image1, x1, y1)
    c2 = show_interest_points(image2, x2, y2)

    # Create the outputImages folder if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputImages")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #save but do not plot the images
    plt.imshow(c1)
    plt.title('Interest Points Image 1')
    plt.savefig(os.path.join(output_dir, "keypointsImage1.png"))
    plt.close()

    plt.imshow(c2)
    plt.title('Interest Points Image 2')
    plt.savefig(os.path.join(output_dir, "keypointsImage2.png"))
    plt.close()

    print('{:d} corners in image 1, {:d} corners in image 2'.format(len(x1), len(x2)))

    binList = [4, 8, 12, 16, 128]
    mainWindowSizeList = [8, 16, 20]
    smallerWindowSizeList = [2, 4, 8]

    #loop over all the possible combinations
    i = 0

    #nested dictionary to store the combinations of the hyperparmaters and their correpsodning accuracy
    accuracyResults = {}

    for bin in binList:
        for mainWindow in mainWindowSizeList:
            for smallWindow in smallerWindowSizeList:
                try:
                    image_name = f"circles_bin{bin}main{mainWindow}small{smallWindow}.jpg"
                    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputImages/EvalImages")
                    image_path = os.path.join(output_dir, image_name)

                    # Check if the image file already exists or if we have already thrown an error for the given state somehow
                    if file_exists(image_path) or accuracyResults.get(bin, {}).get(mainWindow, {}).get(smallWindow) is not None:
                        print(f"Image {image_name} already exists. Skipping to next try.")
                        #jump to next iteration of loop
                        continue

                    image1_features = get_features(image1_bw, x1, y1, feature_width, mainWindow, smallWindow, bin)
                    image2_features = get_features(image2_bw, x2, y2, feature_width, mainWindow, smallWindow, bin)
                    ###Matching features over images
                    numRegions = (mainWindow // smallWindow) ** 2

                    matches, confidences = match_features(image1_features, image2_features, x1, y1, x2, y2, mainWindow, bin, numRegions)
                    print('{:d} matches from {:d} corners'.format(len(matches), len(x1)))

                    ########visualization ########
                    matches = np.round(matches).astype(int)
                    # num_pts_to_visualize = len(matches)
                    num_pts_to_visualize = 100

                    # Create the outputImages folder if it doesn't exist
                    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputImages/EvalImages")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    c1 = show_correspondence_circles(image1, image2,
                                        x1[matches[:num_pts_to_visualize, 0]], y1[matches[:num_pts_to_visualize, 0]],
                                        x2[matches[:num_pts_to_visualize, 1]], y2[matches[:num_pts_to_visualize, 1]])
                    plt.figure(); plt.imshow(c1)
                    plt.savefig(os.path.join(output_dir, f"circles_bin{bin}main{mainWindow}small{smallWindow}.jpg"))
                    
                    c2 = show_correspondence_lines(image1, image2,
                                        x1[matches[:num_pts_to_visualize, 0]], y1[matches[:num_pts_to_visualize, 0]],
                                        x2[matches[:num_pts_to_visualize, 1]], y2[matches[:num_pts_to_visualize, 1]])
                    plt.figure(); plt.imshow(c2)
                    plt.savefig(os.path.join(output_dir, f"lines_bin{bin}main{mainWindow}small{smallWindow}.jpg"))
                    ### when the groundtruth correspondences are available 

                    # num_pts_to_evaluate = len(matches)
                    num_pts_to_evaluate = 100
                    accuracy, c = evaluate_correspondence(image1, image2, eval_file, scale_factor,
                                            x1[matches[:num_pts_to_evaluate, 0]], y1[matches[:num_pts_to_evaluate, 0]],
                                            x2[matches[:num_pts_to_evaluate, 1]], y2[matches[:num_pts_to_evaluate, 1]])
                    #creating nested dictionaries to store the result for the unqiue combination 
                    if bin not in accuracyResults:
                        accuracyResults[bin] = {}
                    if mainWindow not in accuracyResults[bin]:
                        accuracyResults[bin][mainWindow] = {}
                    accuracyResults[bin][mainWindow][smallWindow] = accuracy

                    plt.figure(); plt.imshow(c)
                    plt.savefig(os.path.join(output_dir, f"eval_bin{bin}main{mainWindow}small{smallWindow}accuracy{accuracy}.jpg"))

                    plt.close()
                #exept any exception to continue running
                except Exception as e:
                    # Code to handle the exception
                    if bin not in accuracyResults:
                        accuracyResults[bin] = {}
                    if mainWindow not in accuracyResults[bin]:
                        accuracyResults[bin][mainWindow] = {}
                    #assign this value for dicitonary to be the speciifc error so we can fix it
                    accuracyResults[bin][mainWindow][smallWindow] = str(e)
                    print(str(e))

    #print out the results from the dicitionary
    for var1, var2_dict in accuracyResults.items():
        for var2, var3_dict in var2_dict.items():
            for var3, accuracy in var3_dict.items():
                print(f"Accuracy for bin: {var1}, mainWindowSize: {var2}, smallWindowSize: {var3} is {accuracy}")

if __name__ == "__main__":
    main()