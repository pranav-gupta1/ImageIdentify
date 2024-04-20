import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def add_fft_to_generator(generator):
    for x_batch, y_batch in generator:
        fft_batch = np.fft.fft2(x_batch, axes=(1, 2))
        fft_batch = np.fft.fftshift(fft_batch, axes=(1, 2))
        fft_batch = np.abs(fft_batch)
        fft_batch = np.log(fft_batch + 1)
        fft_batch_max = np.max(fft_batch, axis=(1, 2), keepdims=True)
        fft_batch_min = np.min(fft_batch, axis=(1, 2), keepdims=True)
        fft_batch = (fft_batch - fft_batch_min) / (fft_batch_max - fft_batch_min)
        fft_batch = np.mean(fft_batch, axis=-1, keepdims=True)
        yield [x_batch, fft_batch], y_batch

base_dir = '/Users/pranavgupta/Documents/Synopsys_2024/dataset'
test_dir = '/Users/pranavgupta/Documents/Synopsys_2024/dataset1/test'
model_save_path = os.path.join(base_dir, 'IIModelCNNFFT.h5')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False)

test_generator_with_fft = add_fft_to_generator(test_generator)
model = load_model(model_save_path)
layer_name = 'concatenate'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

activations = []
labels = []
for _ in range(min(20, len(test_generator))):
    [X_img, X_fft], y_batch = next(test_generator_with_fft)
    batch_activations = intermediate_layer_model.predict([X_img, X_fft])
    activations.append(batch_activations)
    labels.append(y_batch)

activations = np.concatenate(activations, axis=0)
labels = np.concatenate(labels, axis=0)
pca = PCA(n_components=3)
activations_reduced = pca.fit_transform(activations)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(activations_reduced[:, 0], activations_reduced[:, 1], activations_reduced[:, 2], c=labels, cmap='coolwarm', alpha=0.5)
ax.set_xlabel('Authenticity Score')
ax.set_ylabel('Texture/Edge Artifacts')
ax.set_zlabel('Color Distribution/Consistency')
ax.set_title('3D PCA of Convolutional/FFT Layer Activations')

plt.show()