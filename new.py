import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

# Step 1: Load and preprocess the data
face_data_dir = 'D:/gender_classification/photos'
handwriting_data_dir = 'D:/gender_classification/signatures'
image_size = (224, 224)  # Input image dimensions for ResNet-50

# Use the Keras ImageDataGenerator to load and preprocess the images
data_generator = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2  # Splitting the data into 80% training and 20% validation
)

# Load and preprocess the data for the face images
face_train_data = data_generator.flow_from_directory(
    face_data_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='binary',  # Assuming binary gender classification
    subset='training'
)

face_valid_data = data_generator.flow_from_directory(
    face_data_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Load and preprocess the data for the handwriting images
handwriting_train_data = data_generator.flow_from_directory(
    handwriting_data_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='binary',  # Assuming binary gender classification
    subset='training'
)

handwriting_valid_data = data_generator.flow_from_directory(
    handwriting_data_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Step 2: Build the ResNet-50 model
# Define and compile the model for face images
base_model_face = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
x = base_model_face.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions_face = layers.Dense(1, activation='sigmoid')(x)
model_face = Model(inputs=base_model_face.input, outputs=predictions_face)

# Freeze the layers of the ResNet-50 base model
for layer in base_model_face.layers:
    layer.trainable = False

model_face.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define and compile the model for handwriting images
base_model_handwriting = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
x = base_model_handwriting.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions_handwriting = layers.Dense(1, activation='sigmoid')(x)
model_handwriting = Model(inputs=base_model_handwriting.input, outputs=predictions_handwriting)

# Freeze the layers of the ResNet-50 base model
for layer in base_model_handwriting.layers:
    layer.trainable = False

model_handwriting.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the models
epochs = 20
model_face.fit(face_train_data, epochs=epochs, validation_data=face_valid_data)
model_handwriting.fit(handwriting_train_data, epochs=epochs, validation_data=handwriting_valid_data)

# Step 4: Evaluate the models
loss_face, accuracy_face = model_face.evaluate(face_valid_data)
loss_handwriting, accuracy_handwriting = model_handwriting.evaluate(handwriting_valid_data)

print("Face Model:")
print(f"Validation Loss: {loss_face:.4f}")
print(f"Validation Accuracy: {accuracy_face*100:.2f}%")

print("\nHandwriting Model:")
print(f"Validation Loss: {loss_handwriting:.4f}")
print(f"Validation Accuracy: {accuracy_handwriting*100:.2f}%")

# Save the models
model_face.save('face_gender_classification_model_resnet50.h5')
model_handwriting.save('handwriting_gender_classification_model_resnet50.h5')
