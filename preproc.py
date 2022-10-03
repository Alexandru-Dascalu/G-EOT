import random


def get_index_generator(size):
    """
    Creates generator for generating indexes of the dataset of 3D models, in random order. It will randomly shuffle
    a list of ints from 0 to size-1, then return one of the numbers at each iteration, until it runs out, and shuffles
    the numbers again, and so on.

    Parameters
    ----------
    size : int
        Number of 3D models in the dataset.

    Returns
    -------
    tensor
        The output tensor.
    """
    perm = list(range(size))
    random.shuffle(perm)

    index = 0
    while True:
        # check if it has reached the end of the epoch
        if index >= size: 
            index = 0
            perm = list(range(size))
            random.shuffle(perm)
        yield perm[index]
        index += 1
