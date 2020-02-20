# CSE140: Wheres Waldo
#

import os
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from keras.models import Model
from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

base_dir = '.'
#train_dir = os.path.join(base_dir,'train/64/train_data_64')
#validation_dir = os.path.join(base_dir, 'train/64/valid_data_64')

train_dir = os.path.join(base_dir,'train/128/train_data_128')
validation_dir = os.path.join(base_dir, 'train/128/valid_data_128')

#train_dir = os.path.join(base_dir,'train/256/train_data_256')
#validation_dir = os.path.join(base_dir, 'train/256/valid_data_256')



# dimensions of our images.
img_width, img_height = 150, 150
img_input = Input(shape=(150,150,3))

mat = Conv2D(filters=4, kernel_size=(3,3))(img_input)
pool = MaxPooling2D((2,2))(mat)

mat2 = Conv2D(filters=5, kernel_size=(3,3))(pool)
pool2 = MaxPooling2D((2,2))(mat2)

mat3 = Conv2D(filters=10, kernel_size=(3,3))(pool2)
pool3 = MaxPooling2D((2,2))(mat3)

tens1d = Flatten()(pool3)

x1 = Dense(20, activation='relu')(tens1d)

output = Dense(1, activation='sigmoid')(x1)

model = Model(img_input, output)

model.summary()

model.compile(loss = 'binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size = (150, 150),  # All images will be resized to 150x150
        batch_size = 2,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size = (150, 150),
        batch_size = 2,
        class_mode = 'binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch = 10,  # 2000 images = batch_size * steps
      epochs = 15,
      validation_data = validation_generator,
      validation_steps = 10,  # 1000 images = batch_size * steps
      verbose = 2)
      
#train_loss, train_acc = mode,evaluate(data, labels)
#test_loss, test_acc = model.evaluate(test_data, test_labels)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
      
print("Training set accuracy:", acc, "\n")
print("validation set accuracy:", val_acc,"\n")
print("Training loss", loss, "\n")
print("Validation loss", val_loss,"\n")
#if K.image_data_format() == 'channels_first':
#    input_shape = (3, img_width, img_height)
#else:
#    input_shape = (img_width, img_height, 3)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Number of epochs')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
