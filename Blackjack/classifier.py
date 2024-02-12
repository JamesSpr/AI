import tensorflow as tf

from keras import layers
from keras import models
from keras import optimizers
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os

# Classifier Settings
img_width, img_height = 250, 200
# optimizers = [optimizers.SGD(learning_rate=0.001), optimizers.Adam(learning_rate=0.001), optimizers.RMSprop(learning_rate=0.001)] 
optimizer = optimizers.SGD(learning_rate=0.0001)
# optimizer = optimizers.Adam(learning_rate=0.001)
# optimizer = optimizers.RMSprop(learning_rate=0.001)
metrics = ['accuracy']
epochs = 8


def create_model():
    #Layers
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(img_width, img_height, 3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(32, (3,3), activation = 'relu'))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(53, activation='softmax'))

    print(model.summary())

    return model

def load_saved_model(model_path="models/example.h5"):

    model = load_model(model_path)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

    return model

def classify_card(card, model):   
    cardVal = ""
    img = card
    im = Image.fromarray(img, 'RGB')

    #Resizing into 128x128 because we trained the model with this image size.
    # im = im.resize((img_height, img_width))
    img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)
    
    result = model.predict(img_array)
    classes = result.argmax(axis=-1)

    # Define Classes
    label_map = {'10C': 0, '10D': 1, '10H': 2, '10S': 3, '2C': 4, '2D': 5, '2H': 6, '2S': 7, '3C': 8, '3D': 9, '3H': 10, '3S': 11, '4C': 12, '4D': 13, '4H': 14, 
                '4S': 15, '5C': 16, '5D': 17, '5H': 18, '5S': 19, '6C': 20, '6D': 21, '6H': 22, '6S': 23, '7C': 24, '7D': 25, '7H': 26, '7S': 27, '8C': 28, '8D': 29, 
                '8H': 30, '8S': 31, '9C': 32, '9D': 33, '9H': 34, '9S': 35, 'AC': 36, 'AD': 37, 'AH': 38, 'AS': 39, 'Back': 40, 'JC': 41, 'JD': 42, 'JH': 43, 'JS': 44, 
                'KC': 45, 'KD': 46, 'KH': 47, 'KS': 48, 'QC': 49, 'QD': 50, 'QH': 51, 'QS': 52 }

    for key, value in label_map.items():
        if classes == value:
            cardVal = key
 
    if len(cardVal) == 3:
        return cardVal[0] + cardVal[1], cardVal[2]
    
    return cardVal[0], cardVal[1]

def test_classification():
    # Test an image for classification
    img = image.load_img('Test1.jpg', target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    prediction = model.predict(img, batch_size=1)
    classes = prediction.argmax(axis=-1)

    label_map = (train_generator.class_indices)

    for key, value in label_map.items():
        if classes == value:
            print("Card is: " + key)

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Define Directory Paths
    base_dir = 'data/dataset'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    validation_dir = os.path.join(base_dir, 'val')

    # Define dataset for training
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height), batch_size=32, class_mode='categorical')
    validation_generator = train_datagen.flow_from_directory(validation_dir, target_size=(img_width, img_height), batch_size=32, class_mode='categorical')

    # Create, Compile and fit the model
    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    # Evaluate model on test data to get accuracy
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height), class_mode='categorical')
    test_loss, test_acc = model.evaluate(test_generator, steps=25)
    predictions = (model.predict(test_generator) > 0.5).astype("int32")

    print("Test Accuracy:", test_acc)

    # Save model
    model.save(f"models/example.h5")