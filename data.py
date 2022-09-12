import os
import csv
import random

import numpy as np
from PIL import Image
from objloader import Obj

import preproc
import uv_renderer
from config import hyper_params

DATA_DIR = "./dataset"

class Model3D:
    def __init__(self, folder, data_dir):
        self.name = folder
        absolute_model_path = os.path.join(data_dir, self.name)

        self.raw_texture = Model3D._get_texture(absolute_model_path)
        self.obj = Obj.open(os.path.join(absolute_model_path, "{}.obj".format(self.name)))
        self.labels = Model3D._load_labels(absolute_model_path)

    def __str__(self):
        return "{}: labels {}".format(self.name, self.labels)

    @staticmethod
    def _get_texture(path):
        """Read texture from file and return it in the appropriate format.

        Parameters
        ----------
        path : String
            Absolute path to dataset sample folder.

        Returns
        -------
        Numpy array
            Numpy array representing the raw texture. Has shape width x height x 3.
        """
        image_path = Model3D._get_texture_path(path)
        texture_image = Image.open(image_path)

        # convert image to a numpy array with float values
        raw_texture = np.array(texture_image).astype(np.float32)
        texture_image.close()
        # some raw textures have an alfa channel too, we only want three colour channels
        raw_texture = raw_texture[:, :, :3]
        # normalise pixel vaues to between 0 and 1
        raw_texture = raw_texture / 255.0

        return raw_texture

    @staticmethod
    def _get_texture_path(path):
        """Determines if texture is a jpg or png file, and returns absolute path to texture file.

        Parameters
        ----------
        path : String
            Absolute path to dataset sample folder.

        Returns
        -------
        String
            Absolute path to texture file.
        """
        if not os.path.isdir(path):
            raise ValueError("The given absolute path is not a directory!")

        for file in os.listdir(path):
            if file.endswith(".jpg"):
                return os.path.join(path, file)
            elif file.endswith(".png"):
                return os.path.join(path, file)

        raise ValueError("No jpg or png files found in the given directory!")

    @staticmethod
    def _load_labels(path):
        """Reads labels of a certain sample from the dataset and returns them.

        Parameters
        ----------
        path : String
            Absolute path to dataset sample folder.

        Returns
        -------
        String or list
            Returns a list of integers, or if this is the dog model, just returns "dog" as a label.
        """
        if not os.path.isdir(path):
            raise ValueError("The given absolute path is not a directory!")

        labels_file_path = os.path.join(path, "labels.txt")
        try:
            labels_file = open(labels_file_path)
        except FileNotFoundError:
            raise FileNotFoundError("No txt files found in the given path! Can not find labels!")

        # labels are written only on the first line of the file, we only read the first line
        labels = next(csv.reader(labels_file, delimiter=','))
        # German shepherd model has all 120+ dog labels as true labels, that is encoded only as "dog" to save
        # make things easier
        if labels[0] == 'dog':
            return labels[0]
        else:
            try:
                int_labels = [int(label) for label in labels]
                return int_labels
            except ValueError as e:
                print("Original exception message: {}".format(str(e)))
                raise ValueError("A label of {} does not represent an int!".format(path))
            finally:
                labels_file.close()


def get_object_folders(data_dir):
    """Returns a list of all folders in the given folder.

    Parameters
    ----------
    path : String
        Absolute path to dataset sample folder.

    Returns
    -------
    List of strings
        Returns a list of with the name of each folder.
    """
    if not os.path.isdir(data_dir):
        raise ValueError("The given data path is not a directory!")

    return [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]


def load_dataset(data_dir):
    object_folders = get_object_folders(data_dir)
    models = [Model3D(folder, data_dir) for folder in object_folders]
    for model in models:
        print(str(model))

    return models


def generate_data_label_pair(models):
    dataset_size = len(models)
    index_generator = preproc.generate_index(dataset_size, shuffle=True)

    while True:
        index = next(index_generator)
        texture = models[index].raw_texture
        obj = models[index].obj
        labels = models[index].labels

        yield texture, obj, labels


def get_adversarial_data_generators(batch_size):
    """ Creates a generator that generate batches of textures from the 3D model dataset, with a random adversarial
    target label.
    Parameters
    ----------
    batch_size : int
        Size of the batches generated by the generators returned by this method.

    Returns
    ----------
    A generator which at each step generates a tuple with three elements: a batch of images, a batch of their true
    labels, and a batch of the desired target labels. The target label is guaranteed to be different than all correct
    labels for that texture.
    """
    models = load_dataset(DATA_DIR)
    renderer = uv_renderer.UVRenderer(None)
    renderer.set_parameters(
        camera_distance=(hyper_params['MinCameraDistance'], hyper_params['MaxCameraDistance']),
        x_translation=(hyper_params['MinTranslationX'], hyper_params['MaxTranslationX']),
        y_translation=(hyper_params['MinTranslationY'], hyper_params['MaxTranslationY'])
    )

    def generate_adversarial_batch():
        generator = generate_data_label_pair(models)

        while True:
            batch_textures = np.zeros(shape=(batch_size, 2048, 2048, 3), dtype=np.float32)
            batch_uv_maps = np.zeros(shape=(batch_size, 299, 299, 2), dtype=np.float32)
            batch_labels = []
            batch_target_labels = []

            for i in range(batch_size):
                batch_textures[i], obj, labels = next(generator)
                batch_labels.append(labels)

                renderer.set_obj(obj)
                batch_uv_maps[i] = renderer.render(i)

                batch_target_labels.append(get_random_target_label(labels))

            batch_target_labels = np.array(batch_target_labels)
            yield batch_textures, batch_uv_maps, batch_labels, batch_target_labels

    return generate_adversarial_batch()


def get_random_target_label(ground_truth_labels):
    label_set = set(ground_truth_labels)

    # loop until we have a random target label distinct from the true labels
    while True:
        target_label = random.randint(0, 999)

        # dog model has al 120+ dog breeds as true labels, so we need to check if the label is outside that range
        if ground_truth_labels == "dog":
            if target_label < 151 or target_label > 275:
                return target_label
        # just check that the chosen target is not in the set of true labels
        elif target_label not in label_set:
            return target_label

def is_prediction_true(true_labels, predicted_label):
    if true_labels == "dog":
        # dog model has all 120 dog breed and dog-like animals as true labels
        if 150 < predicted_label < 276:
            return True
    # even if object only has one true label, it is still represented as a list with just one element
    elif type(true_labels) == list:
        if predicted_label in true_labels:
            return True
    else:
        raise ValueError("true labels list for a sample should be either \"dog\" or a list of ints.")

    # if it has not returned so far, then the prediction is incorrect
    return False


if __name__ == '__main__':
    load_dataset(DATA_DIR)
