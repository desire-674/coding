import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os


# Define image dimensions and batch size

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 2  # e.g., 'fresh' and 'rotten'

# --- Simulating a small dummy dataset for demonstration ---

print("Creating a dummy dataset for demonstration...")

x_train_dummy = np.random.rand(100, IMG_HEIGHT, IMG_WIDTH, 3).astype('float32')
y_train_dummy = np.random.randint(0, NUM_CLASSES, 100) # Random labels (0 or 1)

x_val_dummy = np.random.rand(20, IMG_HEIGHT, IMG_WIDTH, 3).astype('float32')
y_val_dummy = np.random.randint(0, NUM_CLASSES, 20) # Random labels (0 or 1)

# Convert labels to one-hot encoding if your loss function requires it (e.g., categorical_crossentropy)
# For binary classification with sigmoid output, sparse_categorical_crossentropy is often used,
# and labels remain integer. Here, we'll assume a two-class setup suitable for binary cross-entropy.
# If you have more than 2 classes, use tf.keras.utils.to_categorical and 'categorical_crossentropy'.

y_train_dummy_one_hot = tf.keras.utils.to_categorical(y_train_dummy, num_classes=NUM_CLASSES)
y_val_dummy_one_hot = tf.keras.utils.to_categorical(y_val_dummy, num_classes=NUM_CLASSES)


# Create TensorFlow datasets from dummy data

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_dummy, y_train_dummy_one_hot)) \
    .shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_val_dummy, y_val_dummy_one_hot)) \
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print(f"Dummy training data shape: {x_train_dummy.shape}, labels shape: {y_train_dummy_one_hot.shape}")
print(f"Dummy validation data shape: {x_val_dummy.shape}, labels shape: {y_val_dummy_one_hot.shape}")
print("Dummy dataset created successfully.")

# --- 2. Load Pre-trained Model (MobileNetV2) ---

print("\nLoading pre-trained MobileNetV2 model...")
base_model = MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,  # Important: Exclude the top (classification) layer
    weights='imagenet'  # Use weights pre-trained on ImageNet
)
print("MobileNetV2 loaded.")

# --- 3. Freeze Base Model Layers ---

print("Freezing base model layers...")
base_model.trainable = False
print("Base model layers frozen.")

# --- 4. Add Custom Classification Head ---

print("Adding custom classification head...")
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduces spatial dimensions to a single vector per channel
x = Dense(128, activation='relu')(x) # A dense layer
predictions = Dense(NUM_CLASSES, activation='softmax')(x) # Output layer with softmax for multi-class

# Create the new model

model = Model(inputs=base_model.input, outputs=predictions)
print("Custom classification head added.")

# --- 5. Compile the Model ---

print("Compiling the model...")
model.compile(
    optimizer=Adam(learning_rate=0.001), # Adam optimizer with a small learning rate
    loss='categorical_crossentropy',    # Suitable for one-hot encoded labels
    metrics=['accuracy']
)
print("Model compiled.")

# Display model summary

model.summary()

# --- 6. Train the Model ---
# Train the model using the dummy dataset.

print("\nTraining the model (using dummy data)...")

history = model.fit(
    train_dataset,
    epochs=5,  # Number of training epochs (adjust as needed)
    validation_data=validation_dataset,
    verbose=1
)
print("Model training complete.")

# --- 7. Fine-tuning (Optional but Recommended for Better Performance) ---

print("\nStarting fine-tuning (unfreezing top layers and re-training)...")
base_model.trainable = True

# Unfreeze the base model


print(f"Number of layers in the base model: {len(base_model.layers)}")

# Freeze all layers except the last few (e.g., last 20-30 layers)

fine_tune_at = 100 # Adjust this number based on model architecture; typically unfreeze layers closer to output
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False


# Re-compile the model with a very low learning rate for fine-tuning

model.compile(
    optimizer=Adam(learning_rate=0.00001), # Very low learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("Model re-compiled for fine-tuning.")

# Continue training for a few more epochs

fine_tune_epochs = 5
total_epochs =  5 + fine_tune_epochs
history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1, # Start from where previous training left off
    validation_data=validation_dataset,
    verbose=1
)
print("Fine-tuning complete.")


# --- 8. Evaluate the Model ---

print("\nEvaluating the model on validation data...")
loss, accuracy = model.evaluate(validation_dataset, verbose=1)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")


# --- 9. Make Predictions (Example) ---

print("\nMaking predictions on a sample dummy image...")

# Take a single dummy image from the validation set for prediction
sample_image = x_val_dummy[0]
sample_label = y_val_dummy[0]

# Reshape for model input (add batch dimension)
sample_image_expanded = np.expand_dims(sample_image, axis=0)

# Get predictions
predictions = model.predict(sample_image_expanded)
predicted_class_index = np.argmax(predictions[0])

class_names = ['Fresh', 'Rotten'] # Assuming your 0th class is 'Fresh' and 1st is 'Rotten'

print(f"Sample Image true label (index): {sample_label} ({class_names[sample_label]})")
print(f"Predicted probabilities: {predictions[0]}")
print(f"Predicted Class: {predicted_class_index} ({class_names[predicted_class_index]})")



# --- 10. Save the Model ---
# You can save the trained model for future use
# model.save('rotten_produce_classifier.h5') # Keras HDF5 format
# model.save('rotten_produce_classifier_saved_model') # TensorFlow SavedModel format

print("\nModel training and prediction demonstration complete.")



# -*- coding: utf-8 -*-
"""
Rotten Fruit and Vegetable Identification using Transfer Learning
"""


# Install kagglehub if not already installed
!pip install kagglehub

# Import necessary libraries

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import zipfile
import kagglehub

print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# --- 1. Download and Prepare Dataset ---

print("Downloading dataset from Kaggle...")


# Download latest version of the dataset
# This will download the dataset to a temporary directory in Colab

dataset_path = kagglehub.dataset_download("muhriddinmuxiddinov/fruits-and-vegetables-dataset")
print(f"Dataset downloaded to: {dataset_path}")

# Determine the actual base data directory within the downloaded path

base_data_dir = os.path.join(dataset_path, 'Fruits_Vegetables_Dataset(12000)')

if not os.path.exists(base_data_dir):
    print(f"Error: Expected base data directory not found at '{base_data_dir}'")
    raise Exception("Base data directory not found. Please check the downloaded dataset structure.")

# --- 2. Data Preprocessing and Augmentation with Splits ---

# Configuration for image processing
IMAGE_SIZE = (224, 224) # Standard input size for MobileNetV2
BATCH_SIZE = 32



# Define split ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15 # 15% for validation
TEST_SPLIT = 0.15 # 15% for testing




print("\nSetting up data generators with train/validation split...")


# Data augmentation and preprocessing for training and validation images
train_val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input, # Specific preprocessing for MobileNetV2
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VAL_SPLIT # Use 15% of the data for validation
)



all_classes = []
for category_folder in ['Fruits', 'Vegetables']:
    category_path = os.path.join(base_data_dir, category_folder)
    if os.path.exists(category_path):
        for class_name in os.listdir(category_path):
            if os.path.isdir(os.path.join(category_path, class_name)):
                all_classes.append(class_name)

if not all_classes:
    raise Exception("No valid class folders found within the base data directory.")

NUM_CLASSES = len(all_classes)
print(f"Identified {NUM_CLASSES} classes: {all_classes}")

# Creating generators

try:
    train_generator = train_val_datagen.flow_from_directory(
        base_data_dir, # Point to the directory containing 'Fruits' and 'Vegetables'
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training', # Use 'training' subset
        shuffle=True
    )

    validation_generator = train_val_datagen.flow_from_directory(
        base_data_dir, # Point to the same directory
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation', # Use 'validation' subset
        shuffle=False 
    )

  
    class_names = list(train_generator.class_indices.keys())
    print(f"Classes mapped by generator: {class_names}")

except Exception as e:
    print(f"Error setting up data generators: {e}")
    print("It seems flow_from_directory might not be correctly interpreting the nested structure ('Fruits'/'Vegetables').")
    print("Let's try listing files within the 'Fruits' and 'Vegetables' directories to confirm class names.")

    # Fallback: Manually list files to identify classes if flow_from_directory fails
    all_actual_classes = set()
    for category_folder in ['Fruits', 'Vegetables']:
         category_path = os.path.join(base_data_dir, category_folder)
         if os.path.exists(category_path):
             for class_name in os.listdir(category_path):
                 if os.path.isdir(os.path.join(category_path, class_name)):
                     all_actual_classes.add(class_name)

  

    if all_actual_classes:
        print(f"Classes found by manual inspection: {list(all_actual_classes)}")
    else:
         print("No classes found even with manual inspection. There might be an issue with the base_data_dir.")
         raise Exception("Could not identify classes in the dataset.")




# --- 3. Model Building with Transfer Learning (MobileNetV2) ---
# Leveraging a pre-trained MobileNetV2 model and adding custom layers.

print("\nBuilding model with MobileNetV2 transfer learning...")



# Load the MobileNetV2 model pre-trained on ImageNet, excluding the top classification layer [4, 6]
base_model = MobileNetV2(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                         include_top=False,
                         weights='imagenet')



# Freeze the base model layers to retain learned features [3, 5]
base_model.trainable = False


# Add custom classification layers on top of the base model

x = base_model.output
x = GlobalAveragePooling2D()(x) # Global average pooling to flatten the features [4]
x = Dense(256, activation='relu')(x) # A fully connected layer
x = tf.keras.layers.Dropout(0.5)(x) # Dropout for regularization [7, 3]
predictions = Dense(NUM_CLASSES, activation='softmax')(x) # Output layer for multi-class classification [5]

model = Model(inputs=base_model.input, outputs=predictions)


# Compile the model for the feature extraction phase
LEARNING_RATE = 0.001
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy', # Use 'categorical_crossentropy' for multi-class [3, 5]
              metrics=['accuracy'])

model.summary()



# --- 4. Initial Training (Feature Extraction) ---
# Train only the newly added layers.

print("\nTraining (feature extraction phase)...")
EPOCHS_FEATURE_EXTRACTION = 10 # Adjust as needed

history_feature_extraction = model.fit(
    train_generator,
    epochs=EPOCHS_FEATURE_EXTRACTION,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- 5. Fine-tuning ---

# Unfreeze some layers of the base model and train with a lower learning rate.

print("\nFine-tuning phase...")

# Unfreeze the base model layers [5]

base_model.trainable = True



FINE_TUNE_AT = 100 # Unfreeze layers from this point onwards
for layer in base_model.layers[:FINE_TUNE_AT]: # Corrected slicing
    layer.trainable = False



# Recompile the model with a much lower learning rate for fine-tuning [8]
FINE_TUNE_LEARNING_RATE = LEARNING_RATE / 10 # e.g., 0.0001
model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

EPOCHS_FINE_TUNE = 10 # Additional epochs for fine-tuning
total_epochs = EPOCHS_FEATURE_EXTRACTION + EPOCHS_FINE_TUNE


callbacks = []

history_fine_tune = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history_feature_extraction.epoch[-1] + 1, # Start from where feature extraction left off
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks
)



# Load the best model saved by ModelCheckpoint (if ModelCheckpoint was used)


try:
    # Check if the checkpoint file exists before trying to load
    if os.path.exists('best_model.keras'):
        model = tf.keras.models.load_model('best_model.keras')
        print("\nLoaded best model from checkpoint.")
    else:
        print("\nNo model checkpoint found. Using the last trained model.")
except Exception as e:
    print(f"\nCould not load best model from checkpoint: {e}. Using the last trained model.")



# --- 6. Evaluation ---
print("\nEvaluating the model on the validation set...")



# Evaluate on the validation set as we don't have a separate test set with this generator setup.
loss, accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // BATCH_SIZE)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")


# Generate predictions for detailed evaluation (using validation set)

validation_generator.reset() # Important to reset generator before predicting
predictions = model.predict(validation_generator, steps=validation_generator.samples // BATCH_SIZE + 1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes[:len(predicted_classes)] # Ensure lengths match


# Classification Report [9]
print("\nClassification Report (on Validation Data):")



# Adjust target_names to match the order of classes in the generator
report_class_names = list(validation_generator.class_indices.keys())
print(classification_report(true_classes, predicted_classes, target_names=report_class_names))


# Confusion Matrix [3]

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=report_class_names, yticklabels=report_class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Validation Data)')
plt.show()



# --- 7. Visualization of Training History ---

def plot_history(history_fe, history_ft):
  
    # Combine histories
    acc = history_fe.history['accuracy'] + history_ft.history['accuracy']
    val_acc = history_fe.history['val_accuracy'] + history_ft.history['val_accuracy']
    loss = history_fe.history['loss'] + history_ft.history['loss']
    val_loss = history_fe.history['val_loss'] + history_ft.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=EPOCHS_FEATURE_EXTRACTION - 0.5, color='r', linestyle='--', label='Fine-tune Start')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=EPOCHS_FEATURE_EXTRACTION - 0.5, color='r', linestyle='--', label='Fine-tune Start')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

print("\nPlotting combined training history:")
plot_history(history_feature_extraction, history_fine_tune)


# --- 8. Prediction on a New Image (Example) ---

def predict_single_image(model, image_path, class_names, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array) # Apply same preprocessing

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index] # Corrected indexing for confidence

    predicted_class_name = class_names[predicted_class_index]

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class_name} (Confidence: {confidence:.2f})")
    plt.axis('off')
    plt.show()


print("\nTo predict on a new image:")
print("1. Upload an image to your Colab environment (e.g., drag and drop into the file browser).")
print("2. Replace 'path/to/your/image.jpg' with the actual path to your uploaded image.")
print("Example: predict_single_image(model, '/content/your_image.jpg', class_names, IMAGE_SIZE)")

# Inspect the downloaded dataset path to see its contents
print("Contents of the downloaded dataset path:")
!ls -R {dataset_path}
