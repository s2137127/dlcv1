from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import imageio.v2 as imageio
import os
# from matplotlib.pyplot import
import torch
from skimage.transform import resize
import numpy as np
from mean_iou_evaluate import read_masks



def get_transforms():
    return transforms.Compose([
        # transforms.Resize(512,512)
        transforms.ToTensor(),
        transforms.Normalize(
            #[0.485, 0.456, 0.406],
            #[0.229, 0.224, 0.225]
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        )
    ])


class SegmentationData(Dataset):
    def __init__(self, datatype, transform=None):
        self.filename_img = []
        self.filename_mask = []
        self.transform = transform
        # self.path = '../hw1_data/p2_data'
        if datatype == 'train':
            self.path = '../hw1_data/p2_data/train'
            self.filename_img = sorted([file for file in os.listdir(self.path)
                                 if file.endswith('.jpg')])
            self.filename_mask = sorted([file for file in os.listdir(self.path)
                                  if file.endswith('.png')])
        else:
            self.path = '../hw1_data/p2_data/validation'
            self.filename_img = sorted([file for file in os.listdir(self.path)
                                 if file.endswith('.jpg')])
            self.filename_mask = sorted([file for file in os.listdir(self.path)
                                  if file.endswith('.png')])

    def __len__(self):
        return len(self.filename_img)

    def __getitem__(self, idx):

        mask = imageio.imread(os.path.join(self.path,self.filename_mask[idx]))

        image = imageio.imread(os.path.join(self.path, self.filename_img[idx]))
        return image, mask

    def collate_fn(self, batch):
        ims, target = list(zip(*batch))
        # print(len(batch))
        ims = torch.cat([get_transforms()(im.copy() / 255.)[None] for im in ims]).float().to('cuda')
        masks = np.zeros((len(batch), 512, 512))
        for i in range(len(batch)):
            mask = target[i]
            mask = (mask >= 128)
            mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
            masks[i, mask == 3] = 0  # (Cyan: 011) Urban land
            masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land
            masks[i, mask == 5] = 2  # (Purple: 101) Rangeland
            masks[i, mask == 2] = 3  # (Green: 010) Forest land
            masks[i, mask == 1] = 4  # (Blue: 001) Water
            masks[i, mask == 7] = 5  # (White: 111) Barren land
            masks[i, mask == 0] = 6  # (Black: 000) Unknown
        # masks = torch.tensor(masks).to('cuda')
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to('cuda')

        return ims, ce_masks


def get_dataloaders():
    trn_ds = SegmentationData('train')
    val_ds = SegmentationData('valid')

    trn_dl = DataLoader(trn_ds, batch_size=2, shuffle=True, collate_fn=trn_ds.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=val_ds.collate_fn)

    return trn_dl, val_dl
