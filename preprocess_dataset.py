import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# import image utils
from PIL import Image

# import image processing
import scipy.ndimage as ndi
import scipy

# import image utilities
from skimage.morphology import binary_opening, disk, label, binary_closing

# import image augmentation
from albumentations import CLAHE

import os

# get image by id code
def get_image_by_id_code(dataset_path, id_code):
    '''
    Get PIL image by id code
    INPUT:
        dataset_path - path to the image dataset (train or test)
        id_code - image unique id_code
    RETURNS:
        image - PIL image object created from the image file found by id code
    '''
    image = Image.open(os.path.join(dataset_path, str(id_code) + '.png')).convert('L')

    return image

def reduce_noise(mask):
    '''
    Function to reduce noise using morphological operations.
    INPUT:
        mask - (numpy) mask noise reduction to be applied to
    RETURNS:
        reconstruct_final - (numpy) mask with reduced noise
    '''
    eroded_mask = ndi.binary_erosion(mask)
    reconstruct_mask = ndi.binary_propagation(eroded_mask, mask=mask)
    tmp = np.logical_not(reconstruct_mask)
    eroded_tmp = ndi.binary_erosion(tmp)
    reconstruct_final = np.logical_not(ndi.binary_propagation(eroded_tmp, mask=tmp))

    return reconstruct_final

def create_mask(image, threshold = 145.0):
    '''
    Function to create binary numpy mask.
    INPUT:
        image - PIL image to create mask
        threshold - threshold to apply to the image, all regions with color
        intensities above threshold will be segmented by mask
    OUTPUT:
        mask - (numpy) segmented mask
    '''
    # mask out brighter regions
    mask = (np.asarray(image) > threshold).astype(np.float)

    # use morphological operations to reduce noise
    mask = reduce_noise(mask)

    return mask

def segment_eye_edges(image, threshold = 25):
    '''
    Function to segment eye edges using sobel.
    INPUT:
        image - PIL image to segment edges
        threshold - threshold to apply to the image, all regions with color
        intensities above threshold will be segmented by mask
    OUTPUT:
        sob - (numpy) segmented mask for eye edges
    '''
    eye_mask = (np.asarray(image) > threshold).astype(np.float)

    sx = ndi.sobel(eye_mask, axis=0, mode='constant')
    sy = ndi.sobel(eye_mask, axis=1, mode='constant')
    sob = np.hypot(sx, sy).astype(np.float)
    sob = ndi.gaussian_filter(sob, 10)

    sob = reduce_noise(sob)

    sob = binary_opening(sob, disk(4))

    return sob

def get_optical_disk(mask):
    '''
    Function to segment optical disk from mask.
    INPUT:
        mask - (numpy) for optical disk
    OUTPUT:
        sob - (numpy) segmented mask for eye edges
    '''
    label_im, nb_labels = ndi.label(mask)
    sizes = ndi.sum(mask, label_im, range(nb_labels + 1))
    max_size_label = np.argmax(sizes)
    opt_mask = np.where(label_im == max_size_label, 255, 0)

    return opt_mask

def create_mask_dr(image):
    '''
    Function to segment tussues damaged by diabetic retinopathy.
    INPUT:
        image - original PIL image from the dataset
    OUTPUT:
        sob - (numpy) segmented mask tussues damaged by diabetic retinopathy
    '''
    # apply CLAHE
    #clahe = CLAHE(p=1)
    #image = clahe(image=np.array(image))['image']
    #image = ndi.gaussian_filter(image, 10)

    # get mask
    mask = create_mask(image, threshold = np.quantile(np.asarray(image), 0.95))

    # get edges
    edges = segment_eye_edges(image)

    # remove adges from mask
    mask = ndi.gaussian_filter(np.logical_and(mask, np.logical_not(edges)), 1)

    # get optical disk mask
    opt_disk = get_optical_disk(mask)

    # remove optical disk
    mask = np.logical_and(mask, np.logical_not(opt_disk))

    return mask

def preprocess_data(dataset_path, df, processed_path):
    '''
    Function to preprocess train and test datasets.
    INPUT:
        dataset_path - path to the dataset
        df - dataframe containing id_codes for images
        processed_path - output path for preprocessed images
    OUTPUT:
        None
    '''
    for idx in range(len(df)):
        id_code = df.loc[idx]['id_code']
        image = get_image_by_id_code(dataset_path, id_code)
        mask = create_mask_dr(image)

        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, 'L')
        mask_image.save(processed_path + str(id_code) + '.png')

def main():
    # define paths to train images
    TRAIN_IMG_PATH = "../dataset/train_images/"
    TEST_IMG_PATH = "../dataset/test_images/"
    PROCESSED_TRAIN = "../processed/train_images/"
    PROCESSED_TEST = "../processed/test_images/"

    # load csv files with labels as pandas dataframes
    train = pd.read_csv('../dataset/train.csv')
    test = pd.read_csv('../dataset/test.csv')

    preprocess_data(TRAIN_IMG_PATH, train, PROCESSED_TRAIN)
    preprocess_data(TEST_IMG_PATH, test, PROCESSED_TEST)

if __name__ == '__main__':
    main()
