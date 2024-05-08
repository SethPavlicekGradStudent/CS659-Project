import cv2
import numpy as np
import math

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



