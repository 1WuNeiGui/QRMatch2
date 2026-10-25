import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def make_dataset_from_list(flist):
    with open(flist) as f:
        lines = f.readlines()
        imgs = []
        for line in lines:
            imgs.append(line.split(' ')[0])

        targets = [int(line.split(' ')[1].strip()) for line in lines]
        imgs = np.array(imgs)
        targets = np.array(targets)
    return imgs, targets


class MyDataset(Dataset):
    def __init__(self, root=None, weak_stansform=None, transforms=None, strong_stansform=None, lb_train=False,
                 ulb_train=False, test=False, flist=None):
        self.root = root
        self.lb_train = lb_train
        self.ulb_train = ulb_train
        self.test = test
        self.flist = flist
        self.strong_stansform = strong_stansform
        self.transforms = transforms
        self.pictures_path = []
        self.label = []
        self.weak_stansform = weak_stansform

        self.pictures_path, self.label = make_dataset_from_list(flist)

    def __getitem__(self, idex):
        image_path = self.pictures_path[idex]
        label = self.label[idex]
        image = Image.open(image_path).convert("RGB")
        if self.ulb_train:
            image_s = self.strong_stansform(image)
            # u_s2 is a copy of the sampled u_s view for the AIA/pseudo-label path.
            # DDCL can use u_s directly, unaffected by AIA's in-place edits to the copy.
            image_s2 = image_s.clone()
            image_w = self.weak_stansform(image)
            return image_s, image_s2, image_w, image_path, idex
        elif self.lb_train:
            image = self.weak_stansform(image)
            return image, label, idex
        elif self.test:
            image = self.transforms(image)
            return image, label

    def __len__(self):
        return len(self.pictures_path)
