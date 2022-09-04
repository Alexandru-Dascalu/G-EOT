import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from config import cfg


def render(textures, uv_mappings, print_error_params, photo_error_params, background_colour):
    """Use UV mapping to create batch_seize images with both the normal and adversarial texture, then pass the
    adversarial images as input to the victim model to get logits. UV mapping is the matrix M used to transform
    texture x into the image with rendered object, as explained in the paper.

    Returns
    -------
    Tensor of shape batch_size x 1000, representing the logits obtained by passing the adversarial images as
    input to the victim model.
    """
    print_error_params = [(multiplier, addend) for multiplier, addend in zip(print_error_params[0], print_error_params[1])]
    # create each image in batch from texture one at a time. We do this instead of all at once so that we need less
    # memory (a 12 x 2048 x 2048 x 3 tensor is 600 MB, and we would create multiple ones). We make the first image
    # outside of the loop to initialise the list of new images, and to avoid putting an if statement in the loop
    new_images = create_image(textures[0], uv_mappings[0], print_error_params[0])
    for i in range(1, cfg.batch_size):
        image = create_image(textures[i], uv_mappings[i], print_error_params[i])
        new_images = tf.concat([new_images, image], axis=0)

    # add background colour to rendered images.
    new_images = add_background(new_images, uv_mappings, background_colour)
    # plt.imshow(new_images.numpy()[0])
    # plt.imshow(new_images.numpy()[1])
    # plt.imshow(new_images.numpy()[2])
    # plt.imshow(new_images.numpy()[3])
    # plt.imshow(new_images.numpy()[4])
    # plt.imshow(new_images.numpy()[5])


    # check if we apply random noise to simulate camera noise
    if cfg.photo_error:
        new_images = apply_photo_error(new_images, photo_error_params)
        # plt.imshow(new_images.numpy()[0])
        # plt.imshow(new_images.numpy()[1])
        # plt.imshow(new_images.numpy()[2])
        # plt.imshow(new_images.numpy()[3])
        # plt.imshow(new_images.numpy()[4])
        # plt.imshow(new_images.numpy()[5])

    new_images = general_normalisation(new_images)
    # plt.imshow(new_images.numpy()[0])
    # plt.imshow(new_images.numpy()[1])
    # plt.imshow(new_images.numpy()[2])
    # plt.imshow(new_images.numpy()[3])
    # plt.imshow(new_images.numpy()[4])
    # plt.imshow(new_images.numpy()[5])
    return new_images


def create_image(texture, uv_mapping, print_error_params):
    """Create an image from the given texture using the given UV mapping.

    Parameters
    ----------
    texture : Tensor
        Tensor representing the 2048x2048x3 texture of this 3D model. Texture is must have pixel values between 0
        and 255.
    uv_mapping : numpy array
        A numpy array with shape [image_height, image_width, 2]. Represents the UV mappings for an
        image in the batch. This mappign is used to create the images from the textures.

    Returns
    -------
    tuple
        Two tensors. The first one is of shape num_new_renders x 299 x 299 x 3, representing the images of the new
        renders with the normal texture. The second is of shape num_new_renders x 299 x 299 x 3, representing the images
        of the new renders with the adversarial texture.
    """
    # check if we should add print errors, so that the adversarial texture may be used for a 3D printed object
    # and still be effective
    if cfg.print_error:
        print_multiplier, print_addend = print_error_params
        new_texture = transform(tf.identity(texture), print_multiplier, print_addend)
        new_texture = tf.expand_dims(new_texture, axis=0)
    else:
        # tfa.resampler requires input to be in shape batch_size x height x width x channels, so we insert a new
        # dimension
        new_texture = tf.expand_dims(tf.identity(texture), axis=0)

    # plt.imshow(new_texture.numpy()[0])
    # plt.show()

    # tfa.image.resampler requires the first dimension of UV map to be
    # batch size, so we add an extra dimension with one element
    image_uv_map = tf.expand_dims(uv_mapping, axis=0)

    # use UV mapping to create an images corresponding to an individual render by sampling from the texture
    # Resulting tensors are of shape 1 x image_width x image_height x 3
    new_image = tfa.image.resampler(new_texture, image_uv_map)
    # plt.imshow(new_image.numpy()[0])
    # plt.show()

    return new_image


def add_background(images, uv_mappings, background_colour):
    """Colours the background pixels of the image with a random colour.
    """
    # compute a mask with True values for each pixel which represents the object, and False for background pixels.
    mask = tf.reduce_all(input_tensor=tf.not_equal(uv_mappings, 0.0), axis=3, keepdims=True)

    return set_background(images, mask, background_colour)


def get_background_colours():
    return tf.random.uniform([cfg.batch_size, 1, 1, 3], cfg.background_min, cfg.background_max)


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


def get_print_error_args():
    multiplier = tf.random.uniform(
        [cfg.batch_size, 1, 1, 3],
        cfg.channel_mult_min,
        cfg.channel_mult_max
    )
    addend = tf.random.uniform(
        [cfg.batch_size, 1, 1, 3],
        cfg.channel_add_min,
        cfg.channel_add_max
    )

    return multiplier, addend


def apply_photo_error(images, photo_error_params):
    multiplier, addend, noise = photo_error_params
    images = transform(images, multiplier, addend)
    images += noise

    return images


def get_photo_error_args(images_shape):
    multiplier = tf.random.uniform(
        [images_shape[0], 1, 1, 1],
        cfg.light_mult_min,
        cfg.light_mult_max
    )
    addend = tf.random.uniform(
        [images_shape[0], 1, 1, 1],
        cfg.light_add_min,
        cfg.light_add_max
    )

    gaussian_noise = tf.random.truncated_normal(
        images_shape,
        stddev=tf.random.uniform([1], maxval=cfg.stddev)
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


def general_normalisation(x):
    minimum = tf.reduce_min(input_tensor=x, axis=[1, 2, 3], keepdims=True)
    maximum = tf.reduce_max(input_tensor=x, axis=[1, 2, 3], keepdims=True)

    minimum = tf.minimum(minimum, 0)
    maximum = tf.maximum(maximum, 1)

    return (x - minimum) / (maximum - minimum)