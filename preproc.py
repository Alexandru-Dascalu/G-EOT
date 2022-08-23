import cv2
import random
import numpy as np
import tensorflow as tf


def normalise_images(images):
    # normalise images with 0 to 255 values to -1 to 1
    casted = tf.cast(images, tf.float32)
    standardized = tf.identity(casted / 127.5 - 1.0, name='training_standardized')

    return standardized


def generate_index(size, shuffle=True):
    perm = list(range(size))
    if shuffle:
        random.shuffle(perm)
    index = 0
    while True:
        if index >= size: 
            index = 0
            perm = list(range(size))
            if shuffle:
                random.shuffle(perm)
        yield perm[index]
        index += 1


def random_crop(image, size):
    beginX = random.randint(0, image.shape[0]-size[0])
    beginY = random.randint(0, image.shape[1]-size[1])     
      
    return image[beginX:beginX+size[0], beginY:beginY+size[1], :]
        
def center_crop(image, size):
    beginX = int((image.shape[0]-size[0]) / 2)
    beginY = int((image.shape[1]-size[1]) / 2)
    
    return image[beginX:beginX+size[0], beginY:beginY+size[1], :]


def random_rotate(image, rng=10):
    (h, w) = image.shape[0:2]
    angle = (random.random() - 0.5) * rng * 2
    mat = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)

    return cv2.warpAffine(image, mat, (w, h))


def random_flip(image):
    flipped = image
    if random.random() > 0.5: 
        flipped = cv2.flip(flipped, 1) # Flipped Horizontally

    if random.random() > 0.5: 
        flipped = cv2.flip(flipped, 0) # Flipped Vertically
    
    return flipped


def random_horizontal_flip(image):
    flipped = image
    if random.random() > 0.5: 
        flipped = cv2.flip(flipped, 1) # Flipped Horizontally
    
    return flipped


def random_brightness(image, rng=10):
    b = int((random.random()-0.5) * rng * 2)
    return np.uint8(np.clip(np.int32(image) + b, 0, 255))


def random_contrast(image, mini=0.5, maxi=1.5):
    a = mini + (maxi-mini) * random.random()
    b = 125 * (1 - a)
    return np.uint8(np.clip(a * image + b, 0, 255))


def random_shift(image, rng=4):
    shiftX = random.randint(-rng, rng)
    shiftY = random.randint(-rng, rng)
    return np.roll(np.roll(image, shiftX, axis=0), shiftY, axis=1)
    
    
    
