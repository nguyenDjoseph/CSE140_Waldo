import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense


waldoImages = 'train'
list_Images = os.listdir(waldoImages)
print(list_Images)

nrows = 2
ncols = 2
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index +=4
next_waldo_img = [os.path.join(waldoImages, fname) for fname in list_Images[pic_index-4:pic_index]]
i = 0
for img_path in next_waldo_img:
    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis('off')
    img = mpimg.imread(img_path)
    plt.imshow(img)
    i+=1
plt.show()
    
