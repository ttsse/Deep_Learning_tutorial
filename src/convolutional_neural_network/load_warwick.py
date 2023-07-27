import numpy as np
import imageio
import glob
import subprocess
import os

def load_image():

    if not os.path.exists("WARWICK"):
        print("Extracting WARWICK dataset...")
        subprocess.run('unzip ../../datasets/WARWICK.zip', shell=True, check=True)
        print("Done!")
        
    # Loads the MNIST dataset from png images
 
    NUM_TRAIN = 85
    NUM_TEST = 60 
    # create list of image objects
    test_images = []
    test_labels = []    
    
    for idx in range(1, NUM_TEST+1):
        prefix_test = "WARWICK/Test/image_0" if idx < 10 else "WARWICK/Test/image_"
        prefix_label = "WARWICK/Test/label_0" if idx < 10 else "WARWICK/Test/label_"
        image_path_test = prefix_test + str(idx) + ".png"
        image_path_label = prefix_label + str(idx) + ".png"
        image = imageio.imread(image_path_test)
        test_images.append(image)
        label = imageio.imread(image_path_label)
        test_labels.append(label)  
            
    # create list of image objects
    train_images = []
    train_labels = []    

    for idx in range(1, NUM_TRAIN+1):
        prefix_test = "WARWICK/Train/image_0" if idx < 10 else "WARWICK/Train/image_"
        prefix_label = "WARWICK/Train/label_0" if idx < 10 else "WARWICK/Train/label_"
        image_path_test = prefix_test + str(idx) + ".png"
        image_path_label = prefix_label + str(idx) + ".png"
        image = imageio.imread(image_path_test)
        train_images.append(image)
        label = imageio.imread(image_path_label)
        train_labels.append(label)                  
    
    return train_images, train_labels, test_images, test_labels