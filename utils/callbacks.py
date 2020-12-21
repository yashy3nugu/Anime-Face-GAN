import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
"""
Generates a grid image with PREVIEW_ROWS*PREVIEW_COLS of generated images every epoch with PREVIEW_MARGIN
spacing in between
"""
class save_images(keras.callbacks.Callback):

    def __init__(self,noise,preview_margin,preview_rows,preview_cols,**kwargs):
        super(keras.callbacks.Callback,self).__init__(**kwargs)
        self.noise = noise
        self.preview_margin = preview_margin
        self.preview_rows = preview_rows
        self.preview_cols = preview_cols

    def on_epoch_end(self, epoch, logs=None):
        image_array = np.full((
            self.preview_margin + (self.preview_rows * (64 + self.preview_margin)),
            self.preview_margin + (self.preview_cols * (64 + self.preview_margin)), 3),
            255, dtype=np.uint8)

        generated_images = self.model.generator.predict(self.noise)

        generated_images = 0.5 * generated_images + 0.5

        image_count = 0
        for row in range(self.preview_rows):
            for col in range(self.preview_cols):
                r = row * (64 + 16) + self.preview_margin
                c = col * (64 + 16) + self.preview_margin
                image_array[r:r + 64, c:c + 64] = generated_images[image_count] * 255
                image_count += 1

        output_path = 'Epoch_images'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        filename = os.path.join(output_path, f"train-{epoch+1}.png")
        im = Image.fromarray(image_array)
        im.save(filename)


class checkpoint_callback(keras.callbacks.Callback):
    def __init__(self,**kwargs):
        super(keras.callbacks.Callback, self).__init__(**kwargs)
    def on_epoch_end(self, epoch, logs=None):
        self.model.generator.save_weights("Weights/generator_weights.h5")
        self.model.discriminator.save_weights("Weights/discriminator_weights.h5")
