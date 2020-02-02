from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

plt.ion() # Interactive mode for matplotlib.pyplot

## ALLOWS US TO RESCALE THE IMAGE TO A SPECIFIC PIXEL COUNT
class Rescale(object):
    """Rescale the image in a sample to a given point

    args:
        output_size:(tuple or int): desired output size
"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h>w:
                new_h, new_w = self.output_size *h/w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size*w/h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image' : img}


# RANDOMLY CROPS THE IMAGE, NOT PARTICULARLY USEFUL, BUT SHOWS HOW TO CROP
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image= sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image}


## CONVERTS FROM NUMPY ARRAY TO TENSOR(MATRIX INFO)
class ToTensor(object):
    """ Convert ndarrays in sample to tensors."""
    def __call__(self, sample):
        image = sample['image']
        #swap color axis because:
        #numpy img = H x W x C
        #torch img = C x H x W
        image =image.transpose((2, 0, 1))
        return {'image':torch.from_numpy(image)}



## CLASS THAT USES THE waldoinfo.csv TO GIVE THE IMAGES NAMES
class WaldoDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.img_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.img_frame.iloc[idx, 0])
        image = io.imread(img_name)
        sample = {'image':image}

        if self.transform:
            sample = self.transform(sample)
        return sample

# HELPER FUNCTION, PLEASE KEEP FOR NOW    
def show_im(im):
    plt.imshow(im)


waldo_dataset = WaldoDataset(csv_file='train/waldoinfo.csv', root_dir='train/')


#THIS CODE WILL SHOW ONE PICTURE AT A TIME!
for name in range(len(waldo_dataset)):
    sample = waldo_dataset[name]
    show_im(sample['image'])
    plt.show()
    input("ENTER TO CONTINUE")




"""
THIS CODE SHOW HOW TO CROP THINGS THIS IS USEFULL!!!!
scale = Rescale(2560)
crop = RandomCrop(1280)
composed = transforms.Compose([Rescale(2560), RandomCrop(225)])
fig = plt.figure()
sample = waldo_dataset[13]

for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)
    ax = plt.subplot(1, 3, i+1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_im(transformed_sample['image'])

plt.show()
input("ENTER TO CONTINUE")
""" 




"""
THIS CODE WILL SHOW FOUR  PICTURES AT A TIME!
for i in range(len(waldo_dataset)):
    sample = waldo_dataset[i]
    print(i, sample['image'].shape)
    ax = plt.subplot(1, 4, i+1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_im(sample['image'])

    if i==3:
        plt.show()
        input("ENTER TO CONTINUE")
"""


    



