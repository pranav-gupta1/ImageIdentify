import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

model_save_path = '/Users/pranavgupta/Documents/Synopsys_2024/dataset/IIModelCNNOnly.h5'
model = load_model(model_save_path)

layer_name = 'conv2d_1'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

img_width, img_height = 128, 128
batch_size = 32

test_dir = '/Users/pranavgupta/Documents/Synopsys_2024/dataset1/test'
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

activations = []
labels = []
for _ in range(len(test_generator)):
    X_batch, y_batch = next(test_generator)
    batch_activations = intermediate_layer_model.predict(X_batch)
    activations.append(batch_activations.reshape(batch_activations.shape[0], -1))
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
ax.set_title('3D PCA of Convolutional Layer Activations')

plt.show()