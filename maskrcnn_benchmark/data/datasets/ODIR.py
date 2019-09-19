import os

import torch
import torch.utils.data
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from pyexcel_xls import get_data
from torchvision.transforms import functional as F
from torchvision import transforms as tfs

#from maskrcnn_benchmark.structures.bounding_box import BoxList

class ODIRDataset(torch.utils.data.Dataset):
    LABELS = (
        'normal',
        'diabetes',
        'glaucoma',
        'cataract',
        'AMD',
        'hypertension',
        'myopia',
        'other'
    )

    def __init__(self, data_dir, split, orientation, transforms=None):
        self.root = data_dir
        #self.image_set = os.path.join(data_dir, split)
        self.orientation = orientation
        self.opposite_orientation = 'left' if orientation=='right' else 'right'
        #self.transforms = transforms
        #Will be deprecated TODO
        self.randomrotation = tfs.RandomRotation(15)
        self.color = tfs.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0)
        self.size = 512

        self.imgpath = os.path.join(self.root, 'ODIR-5K_training_crop')

        #with open(self.image_set + '.txt') as f:
        #    self.ids = f.readlines()
        #self.ids = [x.strip("\n") for x in self.ids]

        self.imgs_lst = os.listdir(self.imgpath)

        Anno_file_path = os.path.join(self.root, 'departed.xlsx')
        data = get_data(Anno_file_path)
        self.anno_dict = {}
        for d in data[self.orientation][1:]:
            img_name = d[0]
            if img_name not in self.anno_dict:
                if sum(list(map(float, d[1:9]))) == 0 and img_name in self.imgs_lst:
                    self.imgs_lst.remove(img_name)
                else:
                    self.anno_dict[img_name] = [d[1], d[2], d[3],
                        d[4], d[5], d[6], d[7], d[8]]
            else:
                raise KeyError('Key %s repetition' % img_name)

        for d in data[self.opposite_orientation][1:]:
            img_name = d[0]
            if img_name not in self.anno_dict:
                if sum(list(map(float, d[1:9]))) == 0 and img_name in self.imgs_lst:
                    self.imgs_lst.remove(img_name)
                else:
                    self.anno_dict[img_name] = [d[1], d[2], d[3],
                        d[4], d[5], d[6], d[7], d[8]]
            else:
                raise KeyError('Key %s repetition' % img_name)

        #self.imgs_path = os.path.join(self.root, 'ODIR-5K_training_crop')
        self._imgpath = os.path.join(self.root, 'ODIR-5K_training_crop',
                                     "%s")

    def __getitem__(self, index):
        img_name = self.imgs_lst[index]
        img = Image.open(self._imgpath % img_name).convert("RGB")
        if self.opposite_orientation in img_name:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        target = torch.tensor(list(map(float, self.anno_dict[img_name])), requires_grad=False)

        # Transforms TODO
        #if self.transforms is not None:
        #    img, target = self.transforms(img, target)
        img = self.randomrotation(img)
        img = self.color(img)
        img = img.resize((self.size, self.size), Image.ANTIALIAS)
        img = F.to_tensor(img)

        return img, target, index

    def __len__(self):
        return len(self.imgs_lst)

    def get_img_info(self, index):
        return {"height": self.size, "width": self.size}
        
if __name__ == '__main__':
    d = ODIRDataset(
        '/home/zhongying/reference/segment/maskrcnn-benchmark/datasets/ODIR-5K_training',
        'train',
        'right'
    )
    print(len(d))
    print(d[0], d[1000], d[2900])

    dd=d[109]
    print(d.imgs_lst[dd[2]])
    
    img = dd[0]
    image = img.cpu().clone()
    image = image.squeeze(0)
    unloader = tfs.ToPILImage()
    image = unloader(image)

    import matplotlib.pyplot as plt
    plt.figure("input.jpg")
    plt.imshow(image)

    plt.show()

    #import pdb
    #pdb.set_trace()
