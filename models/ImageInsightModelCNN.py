import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.legacy import Adam

#paths
base_dir = '/Users/pranavgupta/Documents/Synopsys_2024/dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
model_save_path = '/Users/pranavgupta/Documents/Synopsys_2024/models/IIModelCNNOnly.h5'

#parameters
img_width, img_height = 128, 128
batch_size = 32
epochs = 20

# train/test data augmentation
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

#check if the model has already been saved and load summary, else perform ML logic
if os.path.exists(model_save_path):
    model = load_model(model_save_path)
    print("Model loaded successfully.")
    model.summary()
else:
    #CNN architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])

    #compile model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    #callbacks (training optimization)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=0.00001)

    #train model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=test_generator.samples // batch_size,
        verbose=1,
        callbacks=[early_stopping, reduce_lr])

    #save model
    model.save(model_save_path)
    print("Model trained and saved.")

#evaluate model
test_steps = test_generator.samples // batch_size
test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)

#calculate loss, accuracy, precision, recall, and F1-score
y_preds = []
y_trues = []
for _ in range(test_steps):
    X, y = next(test_generator)
    y_pred = model.predict(X, verbose=0)
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