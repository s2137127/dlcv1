#python eval.py /home/alex/桌面/hw01/hw1_data/p2_data/validation output/pred_dir/

import os
import imageio.v2 as imageio
import torch
import numpy as np
import torchvision
from model import FCN32,UNet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import torchvision.utils as vutils
import sys
from os import mkdir
from os.path import isdir, dirname, basename
import time
ckpt_path = '145105_fcn32.pth'

cls_color = {
    0:  torch.tensor([0, 255, 255]),
    1:  torch.tensor([255, 255, 0]),
    2:  torch.tensor([255, 0, 255]),
    3:  torch.tensor([0, 255, 0]),
    4:  torch.tensor([0, 0, 255]),
    5:  torch.tensor([255, 255, 255]),
    6:  torch.tensor([0, 0, 0]),
}
# cls_color = torch.tensor(([0, 255, 255],[255, 255, 0],[255, 0, 255],[0, 255, 0],[0, 0, 255],[255, 255, 255],[0, 0, 0]),device='cuda')
class Test(Dataset):
    def __init__(self, path, transform=None):
        self.filename_img = []
        self.transform = transform
        self.path = path
        self.filename_img = sorted([file for file in os.listdir(self.path)
                             if file.endswith('.jpg')])
        # self.filename_img = path

    def __len__(self):
        return len(self.filename_img)

    def __getitem__(self, idx):
        img_name = self.filename_img[idx]
        image = imageio.imread(os.path.join(self.path, img_name))
        # image = imageio.imread(self.filename_img)
        if self.transform:
            image = self.transform(image)
        # print(basename(img_name))
        # return image, basename(self.filename_img).split('.')[0]
        return image, basename(img_name).split('.')[0]


def main():
    data_path, output = sys.argv[1], sys.argv[2]

    model = FCN32(num_classes=7).to('cuda')
    # model = UNet().to('cuda')
    #model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=7).to('cuda')
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    if not isdir(dirname(output)):
        mkdir(dirname(output))
    val_input_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
            
        )
    ])

    test_set = Test(data_path, transform=val_input_transform)
    test_dl = DataLoader(test_set, batch_size=8, shuffle=False,pin_memory=True)

    with torch.no_grad():


        for (data,name) in tqdm(test_dl):



            data = data.to('cuda')
            score = model(data)

            masks = np.argmax(score.cpu().numpy(), 1)

            mask_pic = np.zeros((masks.shape[0],masks.shape[1], masks.shape[2],3),dtype=np.uint8)
            # print(masks)
            for n in range(masks.shape[0]):
                for i in range(mask_pic.shape[1]):
                    for j in range(mask_pic.shape[2]):
                        mask_pic[n, i, j,:] = cls_color[masks[n, i, j]]

                imageio.imsave(output + str(name[n]) + '.png', mask_pic[n])






if __name__ == '__main__':
    main()
