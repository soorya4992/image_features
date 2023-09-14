# Image Feature Extraction and Similarity Search
The following program extracts feature descriptors from the images and performs a similarity search using various similarity measures. The feature extractions used are color Moments, Histogram of oriented Gradients (HOG), and avg_pool layer feature descriptors, layer 3 feature_descriptors, fully connected layer feature_descriptors of a ResNet pre-trained model. These feature descriptors helps us to distinguish images and find k similar images for a given image.

## Features

- **Color Moments:** Extracts color moments from images.
- **HOG Features:** Computes Histogram of Oriented Gradients (HOG) features.
- **ResNet-AvgPool:** Extracts features from the average pooling layer of a pretrained ResNet-50 model.
- **ResNet-Layer3:** Extracts features from the third layer of a pretrained ResNet-50 model.
- **ResNet-FC:** Extracts features from the fully connected layer of a pretrained ResNet-50 model.

## Getting Started


## Dataset

We used Caltech 101 dataset. You can download the dataset from [Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/).

### Installation

1. Install python version greater than 3.0

2. Install various python libraries that needs to be installed using PIP are are torchvision, numpy, scipy, scikit-learn, Pillow, matplotlib

3. Unzip the files and navigate to code directory containing the code file image_features.py

4. Open a the cmd prompt at this file and run the command python image_features.py 

### Usage

1. The program will ask the user to choose an operation from the given menu. The options are 1. Retrieve the feature descriptors based on an input Image Id. 2. To compute feature descriptors and store them. 3. To retrieve k similar operations.

2. For the 1st choice operation, we retrieve data from the  JSON file where we save all the feature descriptors for all images. For this operation to work properly we should have saved the data in the JSON file previously.

2. The option 2 in the menu computes the color moments, HOG, Resnet 50 feature descriptors and store them in a JSON file named Feature_Descriptors.JSON

3. Once data is saved you can use option 1 to retreive the feature descriptors based on the image Id

4. For operation 3, the program asks user for image_id and k value, once these are passed, the similarity functions are used on each feature descriptors to return similar images with their similarity score.

5. The program will display the most similar images using Matplotlib, so you need to close current image to see the next one.


