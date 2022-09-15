import random


def get_index_generator(size, shuffle=True):
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


def center_crop(image, size):
    beginX = int((image.shape[0]-size[0]) / 2)
    beginY = int((image.shape[1]-size[1]) / 2)
    
    return image[beginX:beginX+size[0], beginY:beginY+size[1], :]
