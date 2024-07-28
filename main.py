import os
import cv2
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from collections import Counter

# Define the patch size and other constants
PATCH_SIZE = 200
NUM_CLASSES = 3
DATA_DIR = "D:/Abdur/Woodchip/batch_2_classes/data"  # Set this to your data directory
CLASSES = ['dry', 'medium', 'wet']

# Function to load data by class
def load_data_by_class(data_dir, classes):
    data = {class_name: [] for class_name in classes}
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            image = cv2.imread(img_path)  # Read the image in RGB
            if image is not None:
                data[class_name].append((img_name, image))
    return data

# Function to generate patches from images
def generate_patches(image, patch_size=PATCH_SIZE):
    patches = []
    for i in range(0, image.shape[0] - patch_size + 1, patch_size):
        for j in range(0, image.shape[1] - patch_size + 1, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return np.array(patches)

# Load data by class
data = load_data_by_class(DATA_DIR, CLASSES)

# Split data into training, validation, and test sets at the image level
seed = 42
train_data = {}
val_data = {}
test_data = {}

for class_name in CLASSES:
    images = data[class_name]
    train_val_images, test_images = train_test_split(images, test_size=0.25, random_state=seed)  # 75% train+val, 25% test
    train_images, val_images = train_test_split(train_val_images, test_size=0.2, random_state=seed)  # 20% val from train+val
    train_data[class_name] = train_images
    val_data[class_name] = val_images
    test_data[class_name] = test_images

# Generate patches and labels for training and validation data
def generate_patches_and_labels(data_dict):
    patches, labels = [], []
    for class_name, images in data_dict.items():
        class_index = CLASSES.index(class_name)
        for img_name, image in images:
            img_patches = generate_patches(image)
            patches.append(img_patches)
            labels.extend([class_index] * len(img_patches))
    return np.vstack(patches), np.array(labels)

# Training and validation data
train_patches, train_labels = generate_patches_and_labels(train_data)
val_patches, val_labels = generate_patches_and_labels(val_data)

# Define the ResNet152V2 model
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(PATCH_SIZE, PATCH_SIZE, 3))
base_model.trainable = True

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(train_patches, train_labels,
          validation_data=(val_patches, val_labels),
          epochs=50,
          batch_size=8,
          callbacks=[early_stopping])

# Save the model
model.save("wood_chip_resnet50v2_classifier.h5")

# Function to predict the class of a whole image based on its patches
def predict_image_class(model, image, patch_size=PATCH_SIZE):
    patches = generate_patches(image, patch_size)
    patch_predictions = model.predict(patches)
    predicted_classes = np.argmax(patch_predictions, axis=1)
    most_common_class = Counter(predicted_classes).most_common(1)[0][0]
    return most_common_class

# Testing phase with voting mechanism
test_image_results = {}
correct_predictions = 0
total_images = 0

for class_name in test_data:
    for img_name, image in test_data[class_name]:
        final_prediction = predict_image_class(model, image)
        test_image_results[img_name] = CLASSES[final_prediction]
        if final_prediction == CLASSES.index(class_name):
            correct_predictions += 1
        total_images += 1
        print(f"Predicted class for the image {img_name}: {CLASSES[final_prediction]}")

# Calculate and print the test accuracy
test_accuracy = correct_predictions / total_images
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Optionally, save the results to a file
with open("test_image_predictions.txt", "w") as file:
    for img_name, predicted_class in test_image_results.items():
        file.write(f"{img_name}: {predicted_class}\n")
