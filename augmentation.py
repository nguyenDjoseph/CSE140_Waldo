# augmenting images
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

num = 1
for items in range(0,20):
    
    img = load_img('train/128/valid_data_128/waldo/normal/waldo_valid_128_' + str(num) + '.jpg')
    x = img_to_array(img)
    x = x.reshape((1, ) + x.shape)
    num+=1
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir='train/128/valid_data_128/waldo/augmented', save_format='jpg'):
        i +=1
        if i>20:
            break
