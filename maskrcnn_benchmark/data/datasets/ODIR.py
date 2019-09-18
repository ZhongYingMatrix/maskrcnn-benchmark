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
        self.image_set = os.path.join(data_dir, split)
        self.orientation = orientation
        self.transforms = transforms
        #Will be deprecated TODO
        self.size = 512

        with open(self.image_set + '.txt') as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]

        Anno_file_path = os.path.join(self.root, 'departed.xlsx')
        data = get_data(Anno_file_path)
        self.anno_dict = {}
        for d in data[self.orientation][1:]:
            id = d[0].split('_')[0]
            if id not in self.anno_dict:
                if sum(list(map(float, d[1:9]))) == 0 and id in self.ids:
                    self.ids.remove(id)
                else:
                    self.anno_dict[id] = [d[1], d[2], d[3],
                        d[4], d[5], d[6], d[7], d[8]]
            else:
                raise KeyError('Key %d repetition' % id)

        #self.imgs_path = os.path.join(self.root, 'ODIR-5K_training_crop')
        self._imgpath = os.path.join(self.root, 'ODIR-5K_training_crop',
                                     "%s.jpg")

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_file = img_id + '_' + self.orientation
        img = Image.open(self._imgpath % img_file).convert("RGB")

        target = torch.tensor(list(map(float, self.anno_dict[img_id])), requires_grad=False)

        # Transforms TODO
        #if self.transforms is not None:
        #    img, target = self.transforms(img, target)
        img = img.resize((self.size, self.size), Image.ANTIALIAS)
        img = F.to_tensor(img)

        return img, target, index

    def __len__(self):
        return len(self.ids)

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
    dd=d[2991]
