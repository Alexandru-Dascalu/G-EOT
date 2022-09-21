import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa


def render(textures, uv_mappings, print_error_params, photo_error_params, background_colour, hyper_params):
    """Use UV mapping to create batch_seize images with both the normal and adversarial texture. UV mapping is the
    matrix M used to transfor texture x into the image with rendered object, as explained in the paper.

    Returns
    -------
    Tensor of shape batch_size x 299 x 299 x x3, representing the images rendered with the given textures and uv
    mappings. The images have pixel values normalised to between 0 and 1.
    """
    new_images = create_images(textures, uv_mappings, hyper_params, print_error_params)

    # add background colour to rendered images.
    new_images = add_background(new_images, uv_mappings, background_colour)
    # plt.imshow(new_images.numpy()[0])
    # plt.imshow(new_images.numpy()[1])
    # plt.imshow(new_images.numpy()[2])
    # plt.imshow(new_images.numpy()[3])
    # plt.imshow(new_images.numpy()[4])
    # plt.imshow(new_images.numpy()[5])

    # check if we apply random noise to simulate camera noise
    if hyper_params['PhotoError']:
        new_images = apply_photo_error(new_images, photo_error_params)
        # plt.imshow(new_images.numpy()[0])
        # plt.imshow(new_images.numpy()[1])
        # plt.imshow(new_images.numpy()[2])
        # plt.imshow(new_images.numpy()[3])
        # plt.imshow(new_images.numpy()[4])
        # plt.imshow(new_images.numpy()[5])

    new_images = normalisation(new_images)
    # plt.imshow(new_images.numpy()[0])
    # plt.imshow(new_images.numpy()[1])
    # plt.imshow(new_images.numpy()[2])
    # plt.imshow(new_images.numpy()[3])
    # plt.imshow(new_images.numpy()[4])
    # plt.imshow(new_images.numpy()[5])
    return new_images


def create_images(textures, uv_mappings, hyper_params, print_error_params=None):
    """Create an image from the given texture using the given UV mapping.

    Parameters
    ----------
    textures : Tensor
        Tensor representing the batch_size x 2048 x 2048 x 3 texture of this 3D model. Texture is must have 32-bit float
        pixel values between 0 and 1.
    uv_mappings : numpy array
        A numpy array with shape batch_size x image_height x image_width x 2. Represents the UV mappings for an
        image in the batch. This mapping is used to create the images from the textures.

    Returns
    -------
    tuple
        Two tensors. The first one is of shape num_new_renders x 299 x 299 x 3, representing the images of the new
        renders with the normal texture. The second is of shape num_new_renders x 299 x 299 x 3, representing the images
        of the new renders with the adversarial texture.
    """
    # check if we should add print errors, so that the adversarial textures may be used for a 3D printed object
    # and still be effective
    if hyper_params['PrintError']:
        print_multiplier, print_addend = print_error_params
        textures = transform(textures, print_multiplier, print_addend)

    # use UV mapping to create an images corresponding to an individual render by sampling from the texture
    # Resulting tensors are of shape batch_size x image_width x image_height x 3
    new_images = tfa.image.resampler(textures, uv_mappings)

    return new_images


def add_background(images, uv_mappings, background_colour):
    """Colours the background pixels of the image with a random colour.
    """
    # compute a mask with True values for each pixel which represents the object, and False for background pixels.
    mask = tf.reduce_all(input_tensor=tf.not_equal(uv_mappings, 0.0), axis=3, keepdims=True)

    return set_background(images, mask, background_colour)


def get_background_colours(hyper_params):
    return tf.random.uniform([hyper_params['BatchSize'], 1, 1, 3], hyper_params['MinBackgroundColour'],
                             hyper_params['MaxBackgroundColour'])


def set_background(x, mask, colours):
    """Sets background color of an image according to a boolean mask.

    Parameters
    ----------
        x: A 4-D tensor with shape [batch_size, height, size, 3]
            The images to which a background will be added.
        mask: boolean mask with shape [batch_size, height, width, 1]
            The mask used for determining where are the background pixels. Has False for background pixels,
            True otherwise.
        colours: tensor with shape [batch_size, 1, 1, 3].
            The background colours for each image
    """
    mask = tf.tile(mask, [1, 1, 1, 3])
    inverse_mask = tf.logical_not(mask)

    return tf.cast(mask, tf.float32) * x + tf.cast(inverse_mask, tf.float32) * colours


def get_print_error_args(hyper_params):
    multiplier = tf.random.uniform(
        [hyper_params['BatchSize'], 1, 1, 3],
        hyper_params['PrintErrorMultMin'],
        hyper_params['PrintErrorMultMax']
    )
    addend = tf.random.uniform(
        [hyper_params['BatchSize'], 1, 1, 3],
        hyper_params['PrintErrorAddMin'],
        hyper_params['PrintErrorAddMax']
    )

    return multiplier, addend


def apply_photo_error(images, photo_error_params):
    multiplier, addend, noise = photo_error_params
    images = transform(images, multiplier, addend)
    images += noise

    return images


def get_photo_error_args(hyper_params):
    multiplier = tf.random.uniform(
        [hyper_params['BatchSize'], 1, 1, 1],
        hyper_params['PhotoErrorMultMin'],
        hyper_params['PhotoErrorMultMax']
    )
    addend = tf.random.uniform(
        [hyper_params['BatchSize'], 1, 1, 1],
        hyper_params['PhotoErrorAddMin'],
        hyper_params['PhotoErrorAddMax']
    )

    images_shape = [hyper_params['BatchSize']] + hyper_params['ImageShape'] + [3]
    gaussian_noise = tf.random.truncated_normal(
        images_shape,
        stddev=tf.random.uniform([1], maxval=hyper_params['GaussianNoiseStdDev'])
    )

    return multiplier, addend, gaussian_noise


def transform(x, a, b):
    """Apply transform a * x + b element-wise.

     Parameters
    ----------
        x : tensor
        a : tensor
        b : tensor
    """
    return tf.add(tf.multiply(a, x), b)


def normalisation(x):
    minimum = tf.reduce_min(input_tensor=x, axis=[1, 2, 3], keepdims=True)
    maximum = tf.reduce_max(input_tensor=x, axis=[1, 2, 3], keepdims=True)

    minimum = tf.minimum(minimum, 0)
    maximum = tf.maximum(maximum, 1)

    return (x - minimum) / (maximum - minimum)