import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

#cifare10 is a well name canadian dataset of 60000 32x32 pixel images of the 10 classes of data mentioned below
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifare10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

