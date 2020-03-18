from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2

directory = os.path.dirname(__file__)
test_path = os.path.join(directory, '../data/Test')

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size = (64,64),
    batch_size = 1,
    class_mode = 'binary',
    shuffle = True,
    color_mode = 'rgb',
    classes=['notwaldo','waldo'])

filepath = os.path.join(directory, '../saved_models/trained_model.hdf5')
model=load_model(filepath)

print(model.metrics_names)
print(model.evaluate_generator(test_generator,steps=60))

plt.figure()
for i in range(2):
    for j in range(4):
        img = test_generator.next()
        test_x = np.array(img[0])
        prediction = model.predict(test_x)
        plt.subplot(2, 4, i*4+j+1)
        plt.imshow(np.array(test_x[0, :, :, 0]), cmap='gray')
        predict_val_rounded = round(prediction[0][0], 4)
        plt.title("Prediction " + str(predict_val_rounded))
plt.show()
