from torch.utils.data import Dataset
from PIL import Image
import os


class ImageFiles(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.files = list(os.listdir(self.root))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = os.path.join(self.root, self.files[idx])
        img = Image.open(img_file)
        if self.transform is not None:
            img = self.transform(img)
        return img
