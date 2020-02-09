# CSE140: Wheres Waldo
#
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input


waldoImages = 'train/64/containswaldo'
list_Images = os.listdir(waldoImages)
print(list_Images)

nrows = 2
ncols = 2
pic_index = 0

fig = plt.gcf()

pic_index +=4
next_waldo_img = [os.path.join(waldoImages, fname) for fname in list_Images[1:2]]
i = 0
for img_path in next_waldo_img:
    img = mpimg.imread(img_path)
#    input_layer = Input(shape=(img.size))
#    x1 =  Dense(3, activation='relu')(input_layer)
#    x2 = Dense(2, activation='relu')(x1)
#    x3 = Dense(1, activation='sigmoid')(x2)
    sp = plt.subplot()
    sp.axis('off')
    plt.imshow(img)
    
    i+=1
    plt.show()
    
