import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from keras import models

filenames = os.listdir('/Users/azevedogomes/Documents/A.I./5328327_288472415_Chatbot1/input_dataset')

# create categories for the input image data
categories = []
for filename in filenames:
    # split the filename at the fullstop
    category = filename.split(".")[0]
    if category == 'beer':
        categories.append(0)
    elif category == 'wine':
        categories.append(1)
    elif category == 'spirits':
        categories.append(2)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories,
})
# check the output of the dataframe
# df.head()

# building the image classification model
Image_Classification_Model = Sequential()

Image_Classification_Model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
Image_Classification_Model.add(MaxPooling2D(pool_size=(2, 2)))
Image_Classification_Model.add(Dropout(0.25))

Image_Classification_Model.add(Conv2D(64, (3,3), activation='relu'))
Image_Classification_Model.add(MaxPooling2D(pool_size=(2,2)))
Image_Classification_Model.add(Dropout(0.25))

Image_Classification_Model.add(Flatten())
Image_Classification_Model.add(Dense(96, activation='relu'))
Image_Classification_Model.add(Dropout(0.5))
Image_Classification_Model.add(Dense(3, activation='softmax'))

Image_Classification_Model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# replace the categories with their actual categories
df['category'] = df['category'].replace({0: 'beer', 1: 'wine', 2: 'spirits'})

# splitting the data into train and test sets
df_train, df_validate = train_test_split(df, test_size=0.2, random_state=20)

# reset the indexes of both the train and validation datasets
df_train = df_train.reset_index(drop=True)
df_validate = df_validate.reset_index(drop=True)

total_train = df_train.shape[0]
total_validate = df_validate.shape[0]

# image data augmentation
datagen_train = ImageDataGenerator(
                    rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)
                    rescale=1/255,
                    shear_range=0.1,
                    zoom_range=0.2, # Randomly zoom image
                    horizontal_flip=True,  # randomly flip images
                    width_shift_range=0.1, # randomly shift images horizontally
                    height_shift_range=0.1) # randomly shift images vertically

datagen_validation = ImageDataGenerator(rescale=1/255)


# #  data augmentation
# datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=10,
#         zoom_range = 0.1,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         horizontal_flip=False,  # randomly flip images
#         vertical_flip=False)  # randomly flip images

train_gen = datagen_train.flow_from_dataframe(
                        df_train,
                        'C:/Users/HP/PycharmProjects/Chatbot/input_dataset',
                        x_col = 'filename',
                        y_col = 'category',
                        target_size=(150, 150),
                        class_mode= 'categorical',
                        batch_size=12)

validation_gen = datagen_validation.flow_from_dataframe(df_validate,
                        'C:/Users/HP/PycharmProjects/Chatbot/input_dataset',
                        x_col = 'filename',
                        y_col = 'category',
                        target_size=(150, 150),
                        class_mode= 'categorical',
                        batch_size=12)

epochs = 100
batch_size = 12

# fit the model
history = Image_Classification_Model.fit_generator(train_gen,
                                                   epochs=epochs,
                                                   validation_data=validation_gen,
                                                   validation_steps=total_validate/batch_size,
                                                   steps_per_epoch = len(df_train)/batch_size)

# save the model
Image_Classification_Model.save('ImageClassificationModel.h5')
