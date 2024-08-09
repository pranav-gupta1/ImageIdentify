import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import coremltools

#logging data during training to plot
class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.learning_rates = []
    
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer._decayed_lr(tf.float32)
        if hasattr(lr, 'numpy'):
            lr = lr.numpy()
        self.learning_rates.append(lr)
lr_logger1 = LearningRateLogger()

#find the adjusted information with FFT (feature domain features) analysis applied to it
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

#paths
base_dir = '/Users/pranavgupta/Documents/Synopsys_2024/dataset/'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
model_save_path = '/Users/pranavgupta/Documents/Synopsys_2024/models/IIModelCNNFFT.h5'

#parameters
img_width, img_height = 128, 128
batch_size = 32
epochs = 20

#train/test data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

#train/test data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

#enhanced generators with FFT
train_generator_with_fft = add_fft_to_generator(train_generator)
test_generator_with_fft = add_fft_to_generator(test_generator)

#check if the model has already been saved and load summary, else perform ML logic
if os.path.exists(model_save_path):
    model = load_model(model_save_path)
    print("Model loaded successfully.")
    model.summary()
else:
    input_img = Input(shape=(img_width, img_height, 3), name='img_input')
    input_fft = Input(shape=(img_width, img_height, 1), name='fft_input')

    #CNN architecture on images before FFT
    x = Conv2D(32, (3, 3), activation='relu')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(2, 2)(x)
    x = Flatten()(x)

    #CNN architecture on images after FFT
    y = Conv2D(32, (3, 3), activation='relu')(input_fft)
    y = BatchNormalization()(y)
    y = MaxPooling2D(2, 2)(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D(2, 2)(y)
    y = Conv2D(128, (3, 3), activation='relu')(y)
    y = BatchNormalization()(y)
    y = MaxPooling2D(2, 2)(y)
    y = Flatten()(y)

    #concatenate features from pathways
    combined = concatenate([x, y])

    #fully connected layers
    z = Dense(512, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = BatchNormalization()(z)
    z = Dense(1, activation='sigmoid')(z)

    #model with CNN and FFT integrated
    model = Model(inputs=[input_img, input_fft], outputs=z)

    #compile model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    #callbacks and logging (training optimization)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=0.00001)
    callbacks = [early_stopping, reduce_lr, lr_logger1]

    #train model
    model.fit(
        train_generator_with_fft,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=test_generator_with_fft,
        validation_steps=test_generator.samples // batch_size,
        callbacks=callbacks)

    #save model
    model.save(model_save_path)
    print("Model trained and saved.")

#evaluate model
test_steps = test_generator.samples // batch_size
test_loss, test_acc = model.evaluate(test_generator_with_fft, steps=test_steps)

#calculate loss, accuracy, precision, recall, and F1-score
y_preds = []
y_trues = []
for _ in range(test_steps):
    [X_img, X_fft], y = next(test_generator_with_fft)
    y_pred = model.predict([X_img, X_fft], verbose=0)
    y_preds.extend(y_pred >= 0.5)
    y_trues.extend(y)

precision_metric = Precision()
recall_metric = Recall()
precision_metric.update_state(y_trues, y_preds)
recall_metric.update_state(y_trues, y_preds)

precision = precision_metric.result().numpy()
recall = recall_metric.result().numpy()
if (precision + recall) != 0:
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0

print(f'Test loss: {test_loss*100:.2f}%')
print(f'Test accuracy: {test_acc*100:.2f}%')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1_score:.4f}')

#visual the original image and magnitude spectrum of FFT
img_path = '/Users/pranavgupta/Documents/Synopsys_2024/dataset/test/FAKE/0 (2).jpg'
img = image.load_img(img_path, color_mode='grayscale', target_size=(img_width, img_height))
img_array = image.img_to_array(img)
img_array = np.squeeze(img_array)
img_preprocessed = img_array / 255.
fft_result = np.fft.fft2(img_preprocessed)
fft_shifted = np.fft.fftshift(fft_result)
magnitude_spectrum = 20 * np.log(np.abs(fft_shifted) + 1e-8)
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img_preprocessed, cmap='gray')
plt.title('Original Image'), plt.axis('off')
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.axis('off')
plt.show()

#show graph for learning rate optimization
learning_rates = lr_logger1.learning_rates
epochs = range(1, len(learning_rates) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, learning_rates, marker='o', linestyle='-')
plt.title('Learning Rate over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.show()

#convert to CoreML formatting
coreml_model = coremltools.convert(model, 
                                 minimum_deployment_target=coremltools.target.iOS14,
                                  convert_to='neuralnetwork')
coreml_model.save('ImageDetector.mlmodel')