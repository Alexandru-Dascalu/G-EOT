import os
import csv
from PIL import Image
from objloader import Obj

DATA_DIR = "D:\\Informatica\\GAN-EOT\\GAN-EOT\\dataset"


class Model3D:
    def __init__(self, folder, data_dir):
        self.name = folder
        absolute_model_path = os.path.join(data_dir, self.name)

        self.raw_texture = Model3D._get_image(absolute_model_path)
        self.obj = Obj.open(os.path.join(absolute_model_path, "{}.obj".format(self.name)))
        self.labels = Model3D.load_labels(absolute_model_path)

    def __str__(self):
        return "{}: labels {}".format(self.name, self.labels)

    @staticmethod
    def _get_image(path):
        image_path = Model3D._get_image_path(path)

        texture_image = Image.open(image_path)
        # necessary to match UV map, otherwise rendered object will have the wrong texture
        texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
        raw_image = texture_image.tobytes()
        texture_image.close()

        return raw_image

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
    def load_labels(path):
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

if __name__ == '__main__':
    load_dataset(DATA_DIR)
