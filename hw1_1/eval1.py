import sys
import csv

import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob, os
from os import mkdir
from os.path import isdir, dirname, basename
from PIL import Image
from torchvision import models
from  model import MyNet

class Test(Dataset):
    """ dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image = None
        self.label = None

        self.filesname = []

        '''''
        for file in os.listdir(self.root_dir):
            self.images.append(os.path.join(self.root_dir, file))
        '''''
        for i in os.listdir(self.root_dir):
            self.filesname.append(os.path.join(self.root_dir, i))

    def __len__(self):
        return len(self.filesname)

    def __getitem__(self, idx):
        img_name = self.filesname[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, basename(img_name)


if __name__ == "__main__":
    data_path, output = sys.argv[1], sys.argv[2]
    # model = models.vgg19_bn()
    model = models.inception_v3(pretrained=True)
    model.fc.out_features = 50
    model.load_state_dict(torch.load('111721.pth'))
    model.eval()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    # Load data
    trans = transforms.Compose([transforms.Resize(size=(299, 299)),
                                # transforms.Resize(size=(224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                ])

    test_set = Test(data_path, transform=trans)
    # print(test_set)

    print('Length of Testing Set:', len(test_set))
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    # testing
    prediction = []
    with torch.no_grad():
        for batch_idx, (x, (name,)) in enumerate(test_loader):
            if use_cuda:
                x = x.cuda()
            out = model(x)
            _, pred_label = torch.max(out, 1)
            prediction.append((name, pred_label.item()))
    if not isdir(dirname(output)):
        mkdir(dirname(output))
    with open(output, 'w+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        for i in prediction:
            writer.writerow([i[0], i[1]])
