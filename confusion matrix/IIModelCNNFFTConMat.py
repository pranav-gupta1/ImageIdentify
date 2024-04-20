import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

model_save_path = '/Users/pranavgupta/Documents/Synopsys_2024/dataset/IIModelCNNFFT.h5'
model = load_model(model_save_path)
test_dir = '/Users/pranavgupta/Documents/Synopsys_2024/dataset/test'

img_width, img_height = 128, 128
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

test_generator_with_fft = add_fft_to_generator(test_generator)
test_steps_per_epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)
predictions = model.predict(test_generator_with_fft, steps=test_steps_per_epoch)
predicted_classes = np.where(predictions > 0.5, 1, 0).reshape(-1)
true_classes = test_generator.classes[:len(predicted_classes)]

cm = confusion_matrix(true_classes, predicted_classes)
sns.set(font_scale=1.4)
sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g', cmap='Blues')  # font size
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()