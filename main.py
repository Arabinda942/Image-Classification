pip install tensorflow
pip install keras
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset (handwritten digit images)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Convert labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Build a simple convolutional neural network (CNN) model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')


# Get user input for an image
user_input = input("Enter a path to an image file or type 'random' for a random image: ")

if user_input.lower() == 'random':
    # Generate a random image
    random_image = np.random.rand(28, 28, 1).astype('float32')
    random_image = np.expand_dims(random_image, axis=0)
    predicted_class = np.argmax(model.predict(random_image), axis=-1)[0]
    
    plt.imshow(random_image.reshape((28, 28)), cmap='gray')
    plt.title(f'Model Prediction: {predicted_class}')
    plt.show()
else:
    # Load and preprocess the user-provided image
    try:
        if os.path.exists(user_input):
            user_image = tf.keras.preprocessing.image.load_img(user_input, color_mode='grayscale', target_size=(28, 28))
            user_image = tf.keras.preprocessing.image.img_to_array(user_image)
            user_image = user_image.reshape((1, 28, 28, 1)).astype('float32') / 255

            # Make a prediction
            predicted_class = np.argmax(model.predict(user_image), axis=-1)[0]

            # Display the user-provided image and the model prediction
            plt.imshow(user_image.reshape((28, 28)), cmap='gray')
            plt.title(f'Model Prediction: {predicted_class}')
            plt.show()
        else:
            print(f"Error: The file '{user_input}' does not exist.")
    except Exception as e:
        print(f"Error: {e}")