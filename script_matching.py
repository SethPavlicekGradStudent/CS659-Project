import numpy as np
import math

def match_vertebrae_features(features1, features2, dict, x2, y2):
    Distance = np.zeros((features1.shape[0], features1.shape[1], features2.shape[0]))
    Value = [[],[],[],[],[]]
    Hitx = [[],[],[],[],[]]
    Hity = [[],[],[],[],[]]

    for i in range(features1.shape[0]):
        for j in range(features1.shape[1]):
            for k in range(features2.shape[0]):
                Distance[i][j][k] = np.linalg.norm(features1[i][j] - features2[k])

    print(Distance)
    for i in range(features1.shape[0]):
        for j in range(features1.shape[1]):
            index = np.argmin(Distance[i][j])
            Value[i].append(np.min(Distance[i][j]))
            Hitx[i].append(j)
            Hity[i].append(index)

    Xposistions = np.asarray(Hitx).astype(int)
    print(Xposistions)
    Ypositions = np.asarray(Hity).astype(int)
    matches = np.stack((Xposistions,Ypositions), axis = -1)
    confidences = np.asarray(Value)
    print(matches)

    return matches, confidences

def compute_euclidean_distance(hist1, hist2):
    #histogram Eucleidan distance calcualtion
    squared_diff = (hist1 - hist2) ** 2
    sum_squared_diff = np.sum(squared_diff)
    euclidean_distance = np.sqrt(sum_squared_diff)
    
    return euclidean_distance


def match_features(features1, features2, x1, y1, x2, y2):

    Distance = np.zeros((features1.shape[0], features2.shape[0])) # Euclidean distance. 
    Value = [] # distance of matched feature
    Hitx = [] # x-coordinate of matched feature
    Hity = [] # y-coordinate of matched feature

    #loop over the features1, write your code to find the matched feature in features 2 that has minimal disance. 
    for i in range(features1.shape[0]):
        for j in range(features2.shape[0]):
            sum_equils = 0
            for k in range(16):
                hist_index = k*24
                sum_equils += compute_euclidean_distance(features1[i,hist_index:hist_index+24],features2[j,hist_index:hist_index+24])
            Distance[i][j] = sum_equils / 16

    
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

# def compute_euclidean_distance(hist1, hist2):
#     #histogram Eucleidan distance calcualtion
#     squared_diff = (hist1 - hist2) ** 2
#     sum_squared_diff = np.sum(squared_diff)
#     euclidean_distance = np.sqrt(sum_squared_diff)
    
#     return euclidean_distance

# def match_features(features1, features2, x1, y1, x2, y2):
#     # Initialize lists to store matches and confidences
#     matches = []
#     confidences = []

#     for i, feature1 in enumerate(features1):
#         #initialize the min distance and index to invalid
#         minDist = float('inf')
#         minIndex = -1
        
#         for j, feature2 in enumerate(features2):

#             sum_equils = 0
#             for k in range(16):
#                 hist_index = k*120
#                 sum_equils += compute_euclidean_distance(features1[i,hist_index:hist_index+8],features2[j,hist_index:hist_index+8])
#             dist = sum_equils / 16
            
#             if dist < minDist:
#                 minDist = dist
#                 minIndex = j
        
#         minEucledian = np.argmin([compute_euclidean_distance(features2[minIndex], feature) for feature in features1])
        
#         if i == minEucledian: 
#             confidences.append(minDist)
#             matches.append([i, minIndex])

#     matches = np.array(matches)
#     confidences = np.array(confidences)

#     sorted_indices = np.argsort(confidences)
#     matches = matches[sorted_indices]
#     confidences = confidences[sorted_indices]

#     return matches, confidences

##################################################################################################
# correlation functions from assignment 3

def correlation(imageSlice,template):
    flatimageSlice = imageSlice.flatten()
    flatTemplate = template.flatten()
    return np.corrcoef(flatimageSlice,flatTemplate)[0,1]

#Zero-mean correlation
def zmc(imageSlice,template):
    flatimageSlice = imageSlice.flatten()
    flatTemplate = template.flatten()

    flatImageSliceZeroMean = flatimageSlice - np.mean(flatimageSlice)
    flatTemplateZeroMean = flatTemplate - np.mean(flatTemplate)

    flatImageSliceNorm = np.linalg.norm(flatimageSlice)
    flatTemplateNorm = np.linalg.norm(flatTemplate)

    return np.dot(flatImageSliceZeroMean,flatTemplateZeroMean) / (flatImageSliceNorm*flatTemplateNorm)

#Sum Square Difference
def ssd(imageSlice,template):
    return 1 - np.sum((imageSlice-template)**2)

#Normalized Cross Correlation
def ncc(imageSlice,template):
    flatimageSlice = imageSlice.flatten()
    flatTemplate = template.flatten()

    crossCorrelation = np.correlate(flatimageSlice,flatTemplate)
    normilizationFactor = np.sqrt(np.sum(flatimageSlice**2) * np.sum(flatTemplate**2))
    return crossCorrelation / normilizationFactor
##################################################################################################