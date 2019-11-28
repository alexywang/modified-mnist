from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import cv2
import skimage.data as img_data
from skimage import filters
from skimage import color
import heapq

y_train = pd.read_csv('./train_max_y.csv', sep=',')
y_train = list(y_train['Label'])

x_train = np.load('./train_max_x', allow_pickle=True)
x_test = np.load('./test_max_x', allow_pickle=True)


# Display one image
def display(img):
    plt.imshow(img, cmap ='gray'), plt.title('Image')
    plt.xticks([]), plt.yticks([])
    plt.show()


# Check that all dimensions are the same
def check_shape(data):
    shape = None
    shape = data[0].shape
    for i in range(0, len(data)):
        if shape != data[i].shape:
            return False
    return True


# Blur all images in dataset to remove noise (in-place, takes a long time)
def gaussian_blur(data):
    for i in range(len(data)):
        blur = cv2.GaussianBlur(data[i], (5, 5), 0)
        data[i] = blur


# Segmentation, keep only pixels above a certain intensity threshold, set rest to black
def segment_white(data, threshold=220):
    for i in range(len(data)):
        img = data[i]
        data[i] = np.where(img < threshold, 0, img)
    return data

# Keep only the largest connected components for an image
def filter_components(data):
    data = data.astype(np.uint8)
    for i in range(len(data)):
        img = data[i]
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        sizes = stats[1:, -1]
        min_size = heapq.nlargest(nb_components, sizes)[2]

        img2 = np.zeros(output.shape)
        for j in range(0, nb_components-1):
            if sizes[j] >= 1:
                img2[output == j + 1] = 255
        data[i] = img2
    return data


# Preprocess
x_train = segment_white(x_train)
x_train = filter_components(x_train)

x_test = segment_white(x_test)
x_test = filter_components(x_test)

# Check
display(x_test[588])


# Save
np.save('x_train3.npy', x_train, allow_pickle=True)
np.save('x_test3.npy', x_test, allow_pickle=True)

