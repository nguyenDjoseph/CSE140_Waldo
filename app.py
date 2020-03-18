# CSE140: Wheres Waldo
#

import os
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

base_dir = '.'

train_dir = os.path.join(base_dir,'datav2/Train')
validation_dir = os.path.join(base_dir, 'datav2/Test')

model_save_path = os.path.join(base_dir, 'saved_models/trained_model_testv1.hdf5')

img_width, img_height = 150, 150
img_input = Input(shape=(64,64,3))
'''
mat = Conv2D(filters=64, kernel_size=(5,5), padding='valid', activation='relu')(img_input)
#mat2 = Conv2D(filters=8, kernel_size=(3,3), padding='valid', activation='relu')(mat)
pool = MaxPooling2D(pool_size=(2,2))(mat)

drpout = Dropout(0.05)(pool)

mat3 = Conv2D(filters=32, kernel_size=(3,3), padding='valid', activation='relu')(drpout)
mat4 = Conv2D(filters=32, kernel_size=(2,2), padding='valid', activation='relu')(mat3)
pool2 = MaxPooling2D(pool_size=(2,2))(mat4)

drpout1 = Dropout(0.5)(pool2)

mat5 = Conv2D(filters=16, kernel_size=(2,2), padding='valid', activation='relu')(drpout1)
mat6 = Conv2D(filters=8, kernel_size=(2,2), padding='valid', activation='relu')(mat5)
pool3 = MaxPooling2D(pool_size=(2,2))(mat6)

drpout2 = Dropout(0.25)(pool3)

mat7 = Conv2D(filters=32, kernel_size=(2,2), padding='valid', activation='relu')(drpout2)
#mat8 = Conv2D(filters=8, kernel_size=(2,2), padding='valid', activation='relu')(mat7)
pool4 = MaxPooling2D(pool_size=(2,2))(mat7)

drpout4 = Dropout(0.10)(pool4)

tens1d = Flatten()(drpout4)

x1 = Dense(64, activation='relu')(tens1d)

drpout3 = Dropout(0.10)(x1)

output = Dense(1, activation='sigmoid')(drpout3)

model = Model(img_input, output)
'''
#MODEL v2
mat = Conv2D(filters=128, kernel_size=(9,9), padding='same', activation='relu')(img_input)
pool = MaxPooling2D(pool_size=(5,5))(mat)

dropout = Dropout(0.5)(pool)

mat2 = Conv2D(filters=128, kernel_size=(6,6), padding='same', activation='relu')(dropout)
pool2 = MaxPooling2D(pool_size=(3,3))(mat2)

#drpout2 = Dropout(0.5)(pool2)

mat3 = Conv2D(filters=64, kernel_size=(2,2), padding='same', activation='relu')(pool2)
pool3 = MaxPooling2D(pool_size=(2,2))(mat3)

tens1d = Flatten()(pool3)

x1 = Dense(64, activation='relu')(tens1d)
output = Dense(1, activation='sigmoid')(x1)

model = Model(img_input, output)

model.summary()

model.compile(loss = 'binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size = (64, 64),  # All images will be resized to 150x150
        batch_size = 64,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size = (64, 64),
        batch_size = 64,
        shuffle=True,
        class_mode = 'binary')


end_training = ModelCheckpoint(model_save_path, monitor='acc', verbose=1, save_best_only=True, mode='auto')

history = model.fit_generator(
      train_generator,
      steps_per_epoch = 15,  # 2000 images = batch_size * steps
      epochs = 250,
      validation_data = validation_generator,
      validation_steps = 10,  # 1000 images = batch_size * steps
      verbose = 2,
      callbacks=[end_training])

# end of training related

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
      
print("Training set accuracy:", acc, "\n")
print("validation set accuracy:", val_acc,"\n")
print("Training loss", loss, "\n")
print("Validation loss", val_loss,"\n")

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
