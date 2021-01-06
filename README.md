# Anime-Face-GAN
Generating anime faces using a Deep Convolutional Generative Adversarial Network (DCGAN).  
<img src="assets/generated-images.jpg" height=50% width=50%>  

# Dataset used
The dataset is taken from Kaggle over [here](https://www.kaggle.com/soumikrakshit/anime-faces). The data was obtained from www.getchu.com and processed using a face
detector based on the repo https://github.com/nagadomi/lbpcascade_animeface.
The dataset contains images of size 64 by 64 pixels.

(add dataset image)

# Objective 🎯
The objective of the project is to train the discriminator of the DCGAN by backpropagation using the images of the datset.  
The generator then tries to fool the discriminator every epoch.

# Architecture used

The architecture is inspired by the original DCGAN paper. However 'one-sided label smoothing' has been added to prevent the discriminator from overpowering the generator.

### Generator
The generator takes in a 128 dimensional noise vector sampled from a normal distribution of zero mean and unit variance N(0,1).
It is then followed by a Dense layer of 4x4x1024 units and reshaped to (4,4,1024).  
Then a few transposed convolutional layers are followed which then results in an image of size (64,64,3) with pixel values of the range [-1,1]
due to a tanh activation.  
<img src="assets/generator-model.png" width=40%>

### Discriminator
The discriminator is similar to a image classification CNN which takes in an image and outputs the probability of it being real.  
<img src="assets/discriminator-model.png" width=40%>

# Progress
<img src="assets/training-progress.gif">

# Acknowledgements
- The original DCGAN research paper https://arxiv.org/abs/1511.06434
- Tip for label smoothing https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b
- Helper function to save progress images while training https://github.com/jeffheaton/t81_558_deep_learning

# Libraries used
- [NumPy](https://numpy.org/)
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [matplotlib](https://matplotlib.org/api/pyplot_api.html)
