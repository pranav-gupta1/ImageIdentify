import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

base_dir = '/Users/pranavgupta/Documents/Synopsys_2024/dataset'
test_dir = os.path.join(base_dir, 'test')
model_save_path = os.path.join(base_dir, 'IIModelCNNOnly.h5')

img_width, img_height = 128, 128
batch_size = 32

model = load_model(model_save_path)
print("Model loaded successfully.")

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

test_steps = test_generator.samples // batch_size
predictions = model.predict(test_generator, steps=test_steps)
y_pred = np.where(predictions > 0.5, 1, 0).flatten()
y_true = test_generator.classes[:len(y_pred)]

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()