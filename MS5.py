# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths to the dataset
train_dir = '/work/cseguo/halshehri2/879/project/ISL/train'
test_dir = '/work/cseguo/halshehri2/879/project/ISL/test'

# Define model parameters and data augmentation configuration
img_width, img_height = 100, 100
input_shape = (img_width, img_height, 3)
batch_size = 32
epochs = 50

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    brightness_range=[0.7, 1.3]
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Compute class weights for unbalanced datasets
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class weights:", class_weight_dict)

# Define the CNN architecture
model = Sequential([
    Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(256, (3, 3), padding='same', activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.005)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
#model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
lr_scheduler = LearningRateScheduler(lambda epoch: 0.0001 if epoch < 10 else 0.0005 if epoch < 20 else 0.001)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr, lr_scheduler]
)

# Print training and validation loss and accuracy at each epoch
print("Training History:")
for i, data in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss'])):
    epoch, acc, val_acc, loss, val_loss = i+1, *data
    print(f"Epoch {epoch}: Accuracy = {acc:.4f}, Validation Accuracy = {val_acc:.4f}, Loss = {loss:.4f}, Validation Loss = {val_loss:.4f}")

# Evaluate the model performance
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

predictions = model.predict(test_generator, steps=test_generator.samples // batch_size)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes[:len(predicted_classes)]
cm = confusion_matrix(true_classes, predicted_classes, normalize='true')

# Print the confusion matrix and classification report
print("Confusion Matrix:")
print(cm)

report = classification_report(true_classes, predicted_classes, target_names=list(test_generator.class_indices.keys()))
print("Classification Report:")
print(report)


plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=list(test_generator.class_indices.keys()), yticklabels=list(test_generator.class_indices.keys()))
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.jpg")

plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("accuracy_loss.jpg")

ci_lower, ci_upper = stats.norm.interval(0.95, loc=test_accuracy, scale=np.sqrt((test_accuracy * (1 - test_accuracy)) / len(predicted_classes)))
print(f"95% Confidence Interval for test accuracy: {ci_lower:.2f} to {ci_upper:.2f}")

plt.figure(figsize=(8, 6))
confidence_interval = [ci_lower, ci_upper]
plt.barh(["Test Accuracy"], [test_accuracy], xerr=[(test_accuracy-ci_lower, ci_upper-test_accuracy)], color='gray', alpha=0.4)
plt.xlim(0, 1)
plt.title("Test Accuracy with 95% Confidence Interval")
plt.xlabel("Accuracy")
plt.savefig("confidence_interval.jpg")

# Save the final trained model
model.save('final_model.h5')