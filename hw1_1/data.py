from PIL import Image
import  os
from torch.utils.data import Dataset


class Bike(Dataset):
    """ dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image = None
        self.label = None

        self.filesname = []

        for i in os.listdir(self.root_dir):
            self.filesname.append(os.path.join(self.root_dir, i))


    def __len__(self):
        return len(self.filesname)

    def __getitem__(self, idx):
        img_name = self.filesname[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(img_name.split('/')[-1].split("_")[0])