import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

class save_images(keras.callbacks.Callback):
    """
    This is a subclass of the keras.callbacks.Callback class.
    On subclassing it we can specify methods which can be executed while training
    """

    def __init__(self,noise,preview_margin,preview_rows,preview_cols,**kwargs):
        super(keras.callbacks.Callback,self).__init__(**kwargs)
        self.noise = noise
        self.preview_margin = preview_margin
        self.preview_rows = preview_rows
        self.preview_cols = preview_cols

    # overwriting on_epoch_end() helps in executing a custom method when an epoch ends
    def on_epoch_end(self, epoch, logs=None):
        """
               :param cnt: integer, specifying the epoch number
               :param noise: input noise vector from a normal distribution for the generator to generate images
               :param PREVIEW_MARGIN: int, insert margin between images
               :param PREVIEW_ROWS: int, specify number of rows of images
               :param PREVIEW_COLS: int, specify number of columns of images
               :param generator: neural network model to generate images
               :return: image containing a grid of PREVIEW_ROWS*PREVIEW_COLS images to check generator progress
        """

        # Generate a base array upon which images can then be added sequentially
        image_array = np.full((
            self.preview_margin + (self.preview_rows * (64 + self.preview_margin)),
            self.preview_margin + (self.preview_cols * (64 + self.preview_margin)), 3),
            255, dtype=np.uint8)

        # Generate PREVIEW_ROWS*PREVIEW_COLS number of images using the generator model
        generated_images = self.model.generator.predict(self.noise)
        generated_images = 0.5 * generated_images + 0.5 # Convert pixel intensities to the range [0,1]
        """The images need not be converted into the typical [0,255] pixel intensity values because
            the PIL Image module accepts the range [0,1] 
        """


        image_count = 0
        for row in range(self.preview_rows):
            for col in range(self.preview_cols):
                r = row * (64 + 16) + self.preview_margin
                c = col * (64 + 16) + self.preview_margin
                image_array[r:r + 64, c:c + 64] = generated_images[image_count] * 255
                image_count += 1

        # The image array now contains all the images in an array format which can be stored to the disk

        output_path = 'Epoch_images'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        filename = os.path.join(output_path, f"train-{epoch+1}.png")
        im = Image.fromarray(image_array)
        im.save(filename)


# This is a callback which saves/updates the generator and discriminator weights every epoch in a .h5 format
class checkpoint_callback(keras.callbacks.Callback):
    def __init__(self,**kwargs):
        super(keras.callbacks.Callback, self).__init__(**kwargs)
    def on_epoch_end(self, epoch, logs=None):
        self.model.generator.save_weights("Weights/generator_weights.h5")
        self.model.discriminator.save_weights("Weights/discriminator_weights.h5")
