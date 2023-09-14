#Imports needed to run the code
import torchvision
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import PIL
from PIL import Image
from torchvision import models, transforms
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance import minkowski
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr


#Function defined to find Mean
def findMean(cell_array):
    sum = 0.0
    for i in range(len(cell_array)):
        for j in range(len(cell_array[0])):
            sum = sum + cell_array[i][j]
    N = len(cell_array)*len(cell_array[0])
    return sum/N
    
#Function defined to find Standard deviation using previously calculated Mean
def findStandardDeviation(cell_array, mean):
    sum = 0.0
    for i in range(len(cell_array)):
        for j in range(len(cell_array[0])):
            sum = sum + np.float_power((cell_array[i][j] - mean),2)
    N = len(cell_array)*len(cell_array[0])
    return np.float_power((sum/N),1/2.)

#Function defined to find Skewness using previously calculated Standard Deviation
def findSkewness(cell_array , mean):
    sum = 0.0
    
    for i in range(len(cell_array)):
        for j in range(len(cell_array[0])):
            sum = sum + np.float_power((cell_array[i][j] - mean),3)
    N = len(cell_array)*len(cell_array[0])
    #Using a Negative indicator so that skewness can support Negative values
    neg = False
    if sum < 0:
        neg = True
        sum = sum * -1
    skew = np.float_power((sum/N),1/3.)
    if neg:
        skew = -1 * skew
    return skew



#Function defined to Compute Color Moments for an Image
#Here image_id represents the id of the image in the dataset which is passed as 'dataset'
def computeColorMoments(image_id, dataset):
    img, label = dataset[image_id]            #Retrieving the image from the dataset by image_id
    
    #Resizing the image so that every image is scaled to the same size for Feature Descriptor extraction
    #The image is resized to a image of width 300 and height 100 as per the given problem description
    resized_img = img.resize((300,100))
    img_array = np.array(resized_img)         #Converting the image to numpy array to perform mathematical operations
    grid_size = (10,10)                       #Partitioning the image with a grid of size (10,10) which is defined in the problem description
    cell_size = (img_array.shape[0]//grid_size[0], img_array.shape[1]//grid_size[1])     #The cell size calculated based on the grid used for partition
    color_moments = []                        #Intialising an empty array to store the computed color Moments for each cell in the Image 
    for i in range(10):
        for j in range(10):

            #Calculating the height and width of each cell for which the color Moments are calculated
            h1,h2 = i * cell_size[0], (i + 1) * cell_size[0]
            w1,w2 = j * cell_size[1], (j + 1) * cell_size[1]   
            cell = img_array[h1:h2, w1:w2]              #Slicing the intial image array to include only the cell data which can be used for color Moments computation
            moments = []
            for color_channel in range(3):
                channel_values = cell[:,:,color_channel]               #Dividing the array based on color channels R, G, B

                #Mean, Standard Deviation and Skewness Calculations using the custom defined functions for each color channel
                mean = findMean(channel_values)                       
                standard_deviation = findStandardDeviation(channel_values, mean)
                skewness = findSkewness(channel_values , mean)

                #Adding the color Moments of each color channel to a Temporary array
                moments.append(mean)
                moments.append(standard_deviation)
                moments.append(skewness)

            #Adding the computed color Moments of each cell to the final array
            color_moments.append(moments)
            
    color_moments = np.array(color_moments)
    color_moments_feature_descriptor = color_moments.reshape(10,10,3,3)       #Reshaping the computed colorMoments to 10,10,3,3 shape for better understanding. Here 10,10 represents the cells for which color Moments are calculated and in the 3,3 matrix the row represents the colors and columns represent the mean, standard deviation and skewness values
    return color_moments_feature_descriptor
    

#Function defined to compute Histogram of Oriented Gradients
def computeHOG(imageId, dataset): 
    img, label = dataset[imageId]
    img = img.convert("L")                            #Converting the image to gray scale as we are more focussed on the orientation than colors
    resized_image = img.resize((300,100))
    img_array = np.array(resized_image)
    grid_size = (10,10)                               #Partitioning the image with a 10,10 grid
    cell_size = (img_array.shape[0]//grid_size[0], img_array.shape[1]//grid_size[1])
    dx_mask = np.array([[-1, 0, 1]])                  # Filter Mask we use to convolve the image array for horizantal gradient calculation
    dy_mask = dx_mask.T                               # Transposing the horizantal Filter Mask to convolve the image array for vertical gradient calculation
    num_bins = 9                                      # Splitting the orientation into 9 bins each bins composing 40 degrees
    hog_features = np.zeros((10, 10, 9))
    for i in range(10):
        for j in range(10):
            h1,h2 = i * cell_size[0], (i + 1) * cell_size[0]
            w1,w2 = j * cell_size[1], (j + 1) * cell_size[1]
            cell = img_array[h1:h2, w1:w2]                       
            grad_x = convolve2d(cell, dx_mask, mode='same')         # X gradient calculation 
            grad_y = convolve2d(cell, dy_mask, mode='same')         # Y gradient calculation
            magnitude = np.sqrt(grad_x**2 + grad_y**2)              # Using Pythogrus formula to calculate the magnitude of the gradients
            orientation = np.arctan2(grad_y, grad_x)*180/np.pi      # Calculating the orientation of the gradients and converting it to degrees
            orientation[orientation < 0] += 360                     # Converting the negative orientation to be in the range of 0 to 360    
            hist, bin_edges = np.histogram(orientation, bins=num_bins, range=(0, 360), weights = magnitude)       # Mapping a magintude based histogram for each cell with 9 bins based on their orientation
            hog_features[i, j, :] = hist
    
    hog_feature_descriptors = hog_features.flatten()                # Converting the HOG feature descriptors to 1D from 10, 10, 9 to store in JSON file
    return hog_feature_descriptors


#Function defined to retrieve the vectors from different layers of the pre-trained neural architecture : ResNet50 
def computeResNet50Vectors(imageId, dataset):
    image, label = dataset[imageId]
    resnet = models.resnet50(pretrained=True)          # Loading the ResNet50 model
    hook_output_avg_pool = []                          # list to store the ResNet50 avg pool layer output 
    hook_output_layer3 = []                            # list to store the ResNet50 layer-3 layer output 
    hook_output_fc = []                                # list to store the ResNet50 Full Connected layer output 
    

    # Hook defined to capture the feature vectors from the avg_pool layer
    def hook_fn(module, input, output):
        hook_output_avg_pool.append(output)

    # Hook defined to capture the feature vectors from the Layer3
    def hook_fn_layer3(module, input, output):
        hook_output_layer3.append(output)

    # Hook defined to capture the feature vectors from the Fully Connected layer
    def hook_fn_fc(module, input, output):
        hook_output_fc.append(output)
    
    
    # Registering the defined hook as the forward hook on the avg pool layer on the ResNet 50 model
    avgpool_layer = resnet.avgpool
    avgpool_layer.register_forward_hook(hook_fn)
    
    # Registering the defined hook as the forward hook on the layer-3 on the ResNet 50 model
    layer3 = resnet.layer3
    layer3.register_forward_hook(hook_fn_layer3)

    # Registering the defined hook as the forward hook on the fully connected layer on the ResNet 50 model
    fc_layer = resnet.fc
    fc_layer.register_forward_hook(hook_fn_fc)

    # Performing tranform operations on the image to resize it and retrieve the tensors from each layer
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    resnet(image)                # Passing the image to the ResNet Model
    

    #Storing the output of the hooks in the predefined lists
    avgpool_vector = hook_output_avg_pool[0][0]
    layer3_vector = hook_output_layer3[0][0]
    fc_vector = hook_output_fc[0][0]
    
    
    avgpool_vector = avgpool_vector.squeeze().detach().numpy().reshape(1024, 2)           # Converting the vectors from tensor to numpy to perform mathematical operations
    reduced_avg_pool_dimensionality_vector = np.mean(avgpool_vector, axis = 1)            # Dimensional Reduction is perfromed by averaging up the consecutive numbers and reducing the resultant array to 1D
    
    layer3_vector = layer3_vector.detach().numpy()                                        # Converting the vectors from tensor to numpy to perform mathematical operations
    reduced_layer3_dimensionality_vector = np.mean(layer3_vector, axis = (1, 2))          # Dimensional Reduction is perfromed by averaging up the 14, 14 slice and reducing the resultant array to 1D
    
    fc_vector = fc_vector.detach().numpy()                                                            # Converting the vectors from tensor to numpy to perform mathematical operations
    return (reduced_avg_pool_dimensionality_vector, reduced_layer3_dimensionality_vector, fc_vector)  # returning the feature descriptors of all 3 layers in a tuple



# Function to Retreive the K similar images based on the color moment feature descriptors of an image
def getKSimilarImagesForColorMoments(data, input_image_id, k):
    query_color_moments_vector = np.array(data[str(input_image_id)]['colourMoments']).flatten() #Retrieving the color moments of the image from JSON file and reshaping them to 1 D array for similarity calculation
    similarity_scores = {}
    for image_id, image_data in data.items():
        if (int(image_id) == input_image_id) :             # skipping the image if the image is the input image
            continue
        color_moments_descriptor = np.array(image_data['colourMoments']).flatten()    # Retrieving the color moment feature descriptors of every image for similarity calculation

        distance = minkowski(query_color_moments_vector, color_moments_descriptor, 1)   # Using Minkowski's degree 1 similarity measure to find k-similar images
        
        similarity_scores[image_id] = distance  # Storing the similarity score of each image in a dictionary
    
    sorted_similarity_scores = {key: value for key, value in sorted(similarity_scores.items(), key=lambda item: item[1])}  # Sorting the dictionary based on distance measures in the ascending order so that the most similar objects are at top
    
    # Print the top k similar images and their similarity scores
    for image_id, similarity in list(sorted_similarity_scores.items())[:k]:   
        print(" Using Color Moment feature descriptors, the matching Image ID is: ", image_id, " and Similarity Score is: ", similarity)
        similar_image, label = dataset[int(image_id)]
        plt.imshow(similar_image)                   
        plt.show()                              # Displaying the image using matplotlib
    

# Function to Retreive the K similar images based on the Histogram of Oriented Gradients (HOG) feature descriptors for an image
def getKSimilarImagesForHOG(data, input_image_id, k):
    query_HOG_vector = np.array([data[str(input_image_id)]['HOG']])  #Retrieving the Histogram of Oriented Gradients feature descriptors of the image from JSON file and reshaping them to 2D array for similarity calculation
    similarity_scores = {}
    for image_id, image_data in data.items():
        if (int(image_id) == input_image_id):                               # skipping the image if the image is the input image
            continue
        HOG_descriptor = np.array([image_data['HOG']])                 # Retrieving the Histogram of Oriented Gradients feature descriptors of every image and reshaping them to 2D array for similarity calculation
        similarity = cosine_similarity(query_HOG_vector, HOG_descriptor)  # Using Cosines similarity measure to find k-similar images
        similarity_scores[image_id] = similarity[0][0]             # Storing the similarity score of each image in a dictionary
    sorted_similarity_scores = {key: value for key, value in sorted(similarity_scores.items(), key=lambda item: item[1], reverse= True)}   #Sorting according to similarity scores so that highest similar score image is at the top
    
    # Print the top k similar images and their similarity scores
    for image_id, similarity in list(sorted_similarity_scores.items())[:k]:
        print(" Using HOG, the matching Image ID is: ", image_id, " and Similarity Score is: ", similarity)
        similar_image, label = dataset[int(image_id)]
        plt.imshow(similar_image)                     # Displaying the image using matplotlib
        plt.show()
    

# Function to Retreive the K similar images based on the ResNet50 Avg pool feature descriptors for an image
def getKSimilarImagesForRsNetAvgpool(data, input_image_id, k):
    query_avg_pool_vector = np.array(data[str(input_image_id)]['avg_pool'])   #Retrieving the ResNet50 Avg Pool feature descriptors of the image from JSON file for similarity calculation
    similarity_scores = {}
    for image_id, image_data in data.items():
        if (int(image_id) == input_image_id):                           # skipping the image if the image is the input image
            continue
        avg_pool_descriptor = np.array(image_data['avg_pool']).flatten()  # Retrieving the ResNet50 avg pool feature descriptors of every image for similarity calculation
        # Calculate Euclidean distance
        distance = minkowski(query_avg_pool_vector, avg_pool_descriptor, 2)    # Using Minkowski's degree 2 similarity measure to find k-similar images
       
        similarity_scores[image_id] = distance               # Storing the similarity score of each image in a dictionary
    
    sorted_similarity_scores = {key: value for key, value in sorted(similarity_scores.items(), key=lambda item: item[1])}   # Sorting the dictionary based on distance measures in the ascending order so that the most similar objects are at top
    
    # Print the top k similar images and their similarity scores
    for image_id, similarity in list(sorted_similarity_scores.items())[:k]:
        print(" Using Avg Pool Feature Descriptors, the matching Image ID is: ", image_id, " and Similarity Score is: ", similarity)
        similar_image, label = dataset[int(image_id)]
        plt.imshow(similar_image)         # Displaying the image using matplotlib
        plt.show()
    

# Function to Retreive the K similar images based on the ResNet50 Layer 3 feature descriptors for an image
def getKSimilarImagesForRsNetLayer3(data, input_image_id, k):
    query_layer3_vector = np.array(data[str(input_image_id)]['layer3'])   #Retrieving the ResNet50 Layer3 feature descriptors of the image from JSON file for similarity calculation
    similarity_scores = {}
    for image_id, image_data in data.items():
        if (int(image_id) == input_image_id):                     # skipping the image if the image is the input image
            continue
        layer3_descriptor = np.array(image_data['layer3']).flatten()    # Retrieving the ResNet50 layer3 feature descriptors of every image for similarity calculation
        
        distance = minkowski(query_layer3_vector, layer3_descriptor, 2)  # Using Minkowski's degree 2 similarity measure to find k-similar images
        
        similarity_scores[image_id] = distance   # Storing the similarity score of each image in a dictionary
    
    sorted_similarity_scores = {key: value for key, value in sorted(similarity_scores.items(), key=lambda item: item[1])}     # Sorting the dictionary based on distance measures in the ascending order so that the most similar objects are at top
    
    # Print the top k similar images and their similarity scores
    for image_id, similarity in list(sorted_similarity_scores.items())[:k]:
        print(" Using Layer3 Feature Descriptors, The matching Image ID is: ", image_id, " and Similarity Score is: ", similarity)
        similar_image, label = dataset[int(image_id)]
        plt.imshow(similar_image)                    # Displaying the image using matplotlib
        plt.show()
        

# Function to Retreive the K similar images based on the ResNet50 Fully Connected layer feature descriptors for an image
def getKSimilarImagesForRsNetFCLayer(data, input_image_id, k):
    query_FC_vector = np.array([data[str(input_image_id)]['fc']])      #Retrieving the ResNet50 Fully connected layer feature descriptors of the image from JSON file for similarity calculation
    similarity_scores = {}
    for image_id, image_data in data.items():
        if (int(image_id) == input_image_id):                 # skipping the image if the image is the input image
            continue
        FC_descriptor = np.array([image_data['fc']])           # Retrieving the ResNet50 Fully Connected layer feature descriptors of every image for similarity calculation
        
        similarity = cosine_similarity(query_FC_vector, FC_descriptor)  # Using Cosines similarity measure to find k-similar images
        similarity_scores[image_id] = similarity[0][0]     
    
    sorted_similarity_scores = {key: value for key, value in sorted(similarity_scores.items(), key=lambda item: item[1], reverse = True)}
    
    # Print the top k similar images and their similarity scores
    for image_id, similarity in list(sorted_similarity_scores.items())[:k]:
        print(" Using Fully Connected Layer, the matching Image ID is: ", image_id, " and Similarity Score is: ", similarity)
        similar_image, label = dataset[int(image_id)]
        plt.imshow(similar_image)                 # Displaying the image using matplotlib
        plt.show()
    

# Function to load the existing data from a JSON file
def load_data_from_json(json_file):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}  # Initialize with an empty dictionary if the file doesn't exist
    return data

# Function to save the data to a JSON file. If the file is not available it will create a file in the root project
def save_data_to_json(data, json_file):
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)
    


# Main Program 

dataset = torchvision.datasets.Caltech101(r'../Dataset', download = True) # Caltech101 Dataset will be downloaded in the mentioned path, if already downloaded it will not download again 

exit = False     # flag used to know when to terminate the program
json_file_path = '../Feature_Descriptors.json'     # path of the file used to store the data
while(not exit):
    print("Hi there, The following program has 3 menu options. Select an option based on the operation you want to do \n Menu : 1. Get Feature Descriptors for an image Id 2. Compute and Store Feature Descriptors in a JSON file for all the images 3. Get K similar Images for a given imageId \n")  # Menu options for the User to perform operation
    menu_choice = int(input("Enter your choice as 1, 2 or 3 \n"))   # User Input to perform an operation
    if (menu_choice == 1):
        input_image_id = int(input("please enter the image_id for which you want the feature descriptors \n"))   #Image Id input for which the feature descriptors should be printed
        input_image, label = dataset[input_image_id]
        print("The image corresponding to given image_id is")
        plt.imshow(input_image)                 # Displaying the image using matplotlib
        plt.show()
        data = load_data_from_json(json_file_path)
        if (len(data) == 0) :                          # Case when the file has no data
            print("Sorry, the data hasn't been stored yet, Please use choice 2 in the menu to store the data")
        elif ( data.get(str(input_image_id)) is None):      # Case when the image data is not available in the file. Reasons could be the image is a gray scale image or there is no image corresponding to the id given in the input 
            print("Sorry, the Image corresponding to the given image_id seems to be a gray scale image or doesn't exist in the dataset. Please try for some other image")
        else:
            # Printing Feature Descriptors for the given image
            print("Feature Descriptors are as follows:\n",
                 "Colour Moments Feature Descriptors are:", data[str(input_image_id)]['colourMoments'], " \n",
                 "Histogram of Gradient Orientation Feature Descriptors are ", data[str(input_image_id)]['HOG'], " \n",
                 "ResNet50 Avg Pool Layer Feature Descriptors are ", data[str(input_image_id)]['avg_pool'], " \n",
                 "ResNet50 Layer 3 Feature Descriptors are ", data[str(input_image_id)]['layer3'], " \n",
                 "ResNet50 Fully Connected (FC) Layer Feature Descriptors are ", data[str(input_image_id)]['fc'], "\n")
            
    elif (menu_choice == 2):
        # Computing the feature descriptors and storing them in the JSON file
        for image_id in range(len(dataset)):
            img, label = dataset[image_id]
            img_array = np.array(img)
            if (len(img_array.shape) != 3 or img_array.shape[2] != 3):           #Skipping computation of images which doesn't have all 3 colour channels
                continue
            print("Calculating Feature Descriptors for ", image_id)            # Just a print statement to show that program is computing descriptors for the particular image

            #Computation of Feature Descriptors
            colorMomentFD = computeColorMoments(image_id, dataset)
            HOGFeatureDescriptors = computeHOG(image_id, dataset)
            RsNet_output_vectors = computeResNet50Vectors(image_id, dataset)
            data = {}
            data[str(image_id)] = {
                'colourMoments': colorMomentFD.tolist(), 
                'HOG':    HOGFeatureDescriptors.tolist(),        
                'avg_pool': RsNet_output_vectors[0].tolist(),      
                'layer3': RsNet_output_vectors[1].tolist(),         
                'fc': RsNet_output_vectors[2].tolist()           
            }
    
    
        # Saving the updated data to the JSON file
        save_data_to_json(data, json_file_path)
        print("data saved successfully")
    
    elif (menu_choice == 3):
        
        #Searching k similar images
        input_image_id = int(input("Enter the Image Id you want to search for\n"))    # User input for image Id for which similar images has to be retrieved
        k = int(input("Enter the no of similar images you want to search for\n"))     # User input k describing how many similar images should be retrieved
        data = load_data_from_json(json_file_path)            # loading data from the json file
        if (len(data) == 0) :
            print("Sorry, the data hasn't been stored yet, Please use choice 2 in the menu to store the data")    # Case where the data is not saved in the JSON file
        
        elif ( data.get(str(input_image_id)) is None):           # Case when the image data is not available in the file. Reasons could be the image is a gray scale image or there is no image corresponding to the id given in the input          
            print("Sorry, the Image corresponding to the given image_id seems to be a gray scale image or doesn't exist in the dataset. Please try for some other image")
        
        else:
            input_image, label = dataset[input_image_id]
            print("The image corresponding to given image_id is")
            plt.imshow(input_image)                 # Displaying the image using matplotlib
            plt.show()
            # When data exists. We retrieve k similar images using each feature descriptors
            getKSimilarImagesForColorMoments(data, input_image_id, k)
            getKSimilarImagesForHOG(data, input_image_id, k)
            getKSimilarImagesForRsNetAvgpool(data, input_image_id, k)
            getKSimilarImagesForRsNetLayer3(data, input_image_id, k)
            getKSimilarImagesForRsNetFCLayer(data, input_image_id, k)
    
    else:
        #Case where the Menu choice is not one of the accepted ones
        print("Invalid Choice")
    
    exit_choice = int(input(" For Menu enter 1, To exit enter 2 \n"))
    if (exit_choice != 1):    # logic to take input from user on whether to terminate the program or not and implement accordingly.
        exit = True
                   
    

    
    
