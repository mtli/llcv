from os.path import join

from PIL import Image

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms
from pycocotools.coco import COCO


class COCODataset(Dataset):
    def __init__(self, args, is_train=False):
        assert not is_train, 'Training is not implemented'
        self.data_root = args.data_root
        opts = args.data_opts
        self.ann_file = opts['ann_file']

        self.db = COCO(self.ann_file)
        self.iids = self.db.getImgIds()
        # self.classes = [cat['name'] for cat in self.db.cats.values()] + ['background']
        self.classes = 91*[''] # torchvision's detector is trained with extra classes not in the annotation file
        self.pipeline = tv_transforms.ToTensor()

    def __len__(self):
        return len(self.iids)

    def __getitem__(self, idx):
        iid = self.iids[idx]
        file_name = self.db.loadImgs(iid)[0]["file_name"]
        file_path = join(self.data_root, file_name)
        I = Image.open(file_path).convert('RGB')
        I = self.pipeline(I)

        anns = self.db.loadAnns(self.db.getAnnIds(idx))
        target = {'image_id': iid, 'annotations': anns}

        return I, target

