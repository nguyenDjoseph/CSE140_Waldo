from PIL import Image
from keras.models import load_model
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Image file must be in the same directory as main.py")
args = parser.parse_args()
image_name=args.name


directory = os.path.dirname(__file__)
filepath = os.path.join(directory, '../Saved_Models/trained_modelv1.hdf5')
model = tf.keras.models.load_model(filepath)


img = Image.open(image_name)

w = int(img.size[0] / 32) - 1
h = int(img.size[1] / 32) - 1

print(w)
print(h)

bestList = []
result = np.zeros((1, 2))

img2 = np.asarray(img)

cv_img = cv2.imread(image_name, cv2.COLOR_BGR2RGB)
val = []

coordinates = []
for i in range(w):
    for j in range(h):
        cropped_img = cv_img[32*j:32*j+64, 32*i:32*i+64]
        # transform to color image (gray when opencv opens it)
        cropped_img_clr = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        cropped_img_np = np.asarray(cropped_img_clr)
        cropped_img_naxis = cropped_img_np[np.newaxis,:,:,:]/255

        val.append(cropped_img_naxis)

        prediction_val = model.predict(np.asarray(cropped_img_naxis))
        
        if prediction_val[0,0] > 0.95:
            coordinates.append([32*j,32*j+64, 32*i,32*i+64])
            bestList.append(prediction_val[0,0])
            result = np.concatenate((result, np.array([32 * i, 32 * j])[np.newaxis, ...]), 0)

result = result.astype(np.int32)
result = result[1:len(result),:]

toshow = np.concatenate((img2, img2, img2), 2) / 255.
print(coordinates)

for x in coordinates:
    cv_img = cv2.rectangle(cv_img, (x[1],x[3]), (x[0],x[2]), (0,255,0), 2)
    
plt.imshow(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
plt.show()
