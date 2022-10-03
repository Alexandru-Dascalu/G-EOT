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
