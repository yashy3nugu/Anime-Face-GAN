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

    def __init__(self,noise,margin,num_rows,num_cols,**kwargs):
        super(keras.callbacks.Callback,self).__init__(**kwargs)
        self.noise = noise
        self.margin = margin
        self.num_rows = num_rows
        self.num_cols = num_cols

    # overwriting on_epoch_end() helps in executing a custom method when an epoch ends
    def on_epoch_end(self, epoch, logs=None):
        """
        Saves images generated from a fixed random vector by the generator to the disk 
        
        Parameters:
            noise: fixed noise vector from a normal distribution to be fed to the generator.
            num_rows: number of rows of images
            num_cols: number of columns of images
            margin: margin between images
            generator: keras model representing the generator network
        
        """

        # Generate a base array upon which images can then be added sequentially
        image_array = np.full((
            self.margin + (self.num_rows * (64 + self.margin)),
            self.margin + (self.num_cols * (64 + self.margin)), 3),
            255, dtype=np.uint8)

        # Generate num_rows*num_cols number of images using the generator model
        generated_images = self.model.generator.predict(self.noise)

        # Convert pixel intensities to the range [0,1]
        generated_images = 0.5 * generated_images + 0.5

        #Images need not be converted into the typical [0,255] pixel intensity values because the PIL Image module accepts the range [0,1] 


        image_count = 0
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                r = row * (64 + 16) + self.margin
                c = col * (64 + 16) + self.margin
                image_array[r:r + 64, c:c + 64] = generated_images[image_count] * 255
                image_count += 1

        # The image array now contains all the images in an array format which can be stored to the disk

        output_path = 'Epoch_images'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        filename = os.path.join(output_path, f"train-{epoch+1}.png")
        im = Image.fromarray(image_array)
        im.save(filename)


class checkpoint_callback(keras.callbacks.Callback):
    """
    Subclass of keras.callbacks.Callback to save the weights every epoch in a .h5 file
    """
    def __init__(self,**kwargs):
        super(keras.callbacks.Callback, self).__init__(**kwargs)
    def on_epoch_end(self, epoch, logs=None):
        self.model.generator.save_weights("Weights/generator_weights-test.h5")
        self.model.discriminator.save_weights("Weights/discriminator_weights-test.h5")
