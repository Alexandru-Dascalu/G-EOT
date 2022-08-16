import os
from PIL import Image
from objloader import Obj

DATA_DIR = "D:\\Informatica\\GAN-EOT\\GAN-EOT\\dataset"


class Model3D:
    def __init__(self, folder, data_dir):
        self.name = folder

        absolute_model_path = os.path.join(data_dir, self.name)
        self.raw_texture = Model3D._get_image(absolute_model_path)
        self.obj = Obj.open(os.path.join(absolute_model_path, "{}.obj".format(self.name)))

    @staticmethod
    def _get_image(path):
        image_path = Model3D._get_image_path(path)

        texture_image = Image.open(image_path)
        texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
        raw_image = texture_image.tobytes()
        texture_image.close()

        return raw_image

    @staticmethod
    def _get_image_path(path):
        assert os.path.isdir(path)

        for file in os.listdir(path):
            if file.endswith(".jpg"):
                return os.path.join(path, file)
            elif file.endswith(".png"):
                return os.path.join(path, file)

        raise ValueError("No jpg or png files found in the given directory!")


def get_object_folders(data_dir):
    object_paths = [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
    return object_paths


def load_dataset(data_dir):
    object_folders = get_object_folders(data_dir)
    models = [Model3D(folder, data_dir) for folder in object_folders]
    print(models)

if __name__ == '__main__':
    load_dataset(DATA_DIR)
