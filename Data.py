import os
import csv
import random
import numpy as np
from PIL import Image
from objloader import Obj

import Preproc

DATA_DIR = "D:\\Informatica\\GAN-EOT\\GAN-EOT\\dataset"


class Model3D:
    def __init__(self, folder, data_dir):
        self.name = folder
        absolute_model_path = os.path.join(data_dir, self.name)

        self.raw_texture = Model3D._get_image(absolute_model_path)
        self.obj = Obj.open(os.path.join(absolute_model_path, "{}.obj".format(self.name)))
        self.labels = Model3D._load_labels(absolute_model_path)

    def __str__(self):
        return "{}: labels {}".format(self.name, self.labels)

    @staticmethod
    def _get_image(path):
        image_path = Model3D._get_image_path(path)

        texture_image = Image.open(image_path)
        # necessary to match UV map, otherwise rendered object will have the wrong texture
        texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
        raw_image = np.array(texture_image)
        texture_image.close()

        # some raw textures have an alfa channel too, we only want three colour channels
        return raw_image[:, :, :3]

    @staticmethod
    def _get_image_path(path):
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
    index_generator = Preproc.generate_index(dataset_size, shuffle=True)

    while True:
        index = next(index_generator)
        image = models[index].raw_texture
        labels = models[index].labels

        yield image, labels


def get_data_generators(batch_size):
    """ Creates a generator that generate batches of textures from the 3D model dataset.
    Parameters
    ----------
    batch_size : int
        Size of the batches generated by the generators returned by this method.

    Returns
    ----------
    A generator which at each step generates a batch of textures, and a batch of their corresponding ground truth
    labels.
    """
    models = load_dataset(DATA_DIR)

    def generate_texture_batch():
        generator = generate_data_label_pair(models)

        while True:
            batch_textures = []
            batch_labels = []

            for _ in range(batch_size):
                texture, labels = next(generator)
                batch_textures.append(texture)
                batch_labels.append(labels)

            yield batch_textures, batch_labels

    return generate_texture_batch()


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

    def generate_adversarial_batch():
        generator = generate_data_label_pair(models)

        while True:
            batch_textures = []
            batch_labels = []
            batch_target_labels = []

            for _ in range(batch_size):
                texture, labels = next(generator)
                batch_textures.append(texture)
                batch_labels.append(labels)
                batch_target_labels.append(get_random_target_label(labels))

            batch_target_labels = np.array(batch_target_labels)
            yield batch_textures, batch_labels, batch_target_labels

    return generate_adversarial_batch()


def get_random_target_label(ground_truth_labels):
    while True:
        target_label = random.randint(0, 999)

        if ground_truth_labels == "dog":
            if target_label < 151 or target_label > 275:
                return target_label
        elif target_label not in ground_truth_labels:
            return target_label


if __name__ == '__main__':
    load_dataset(DATA_DIR)
