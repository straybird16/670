import torch
from datasets import load_dataset
import numpy as np
import copy

class CustomImageDataSet(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, transform=None):
        super().__init__()
        # to tensors
        images = torch.FloatTensor(images)
        # permute images to shape (N, C, H, W)
        images = images.permute(0,3,1,2)
        if labels is not None:
            labels = torch.LongTensor(labels)

        self.imgs =images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, i):
        if self.labels is None:
            img = self.transform(self.imgs[i])
            return img, copy.deepcopy(img)
        if self.transform is not None:
            return self.transform(self.imgs[i]), self.labels[i]
        return self.imgs[i], self.labels[i]
    
    def __len__(self):
        return len(self.imgs)
    

def load_data(config) -> tuple:
    """
    __args:

    __returns:

    train_x, train_y, val_x, val_y
    """
    if config.dataset == 'cifar100':
        dataset = load_dataset("cifar100")
        train_tag, val_tag = 'train', 'test'
        x_tag, y_tag = 'img', 'fine_label'
       
    elif config.dataset == 'tiny-imagenet':
        dataset = load_dataset("zh-plus/tiny-imagenet")
        train_tag, val_tag = 'train', 'valid'
        x_tag, y_tag = 'image', 'label'

    else:
        raise NotImplementedError("Dataset \"{}\" not recognized.".format(config.dataset))
    
    train_data = dataset[train_tag]
    val_data = dataset[val_tag]
    train_x, train_y = train_data[x_tag], train_data[y_tag]
    val_x, val_y = val_data[x_tag], val_data[y_tag]

    train_x, train_y = filter_gray_images_(train_x, train_y)
    val_x, val_y = filter_gray_images_(val_x, val_y)
    # to 0-1 range
    train_x, train_y, val_x, val_y = train_x/255.0, train_y, val_x/255.0, val_y
    return train_x, train_y, val_x, val_y

    

def filter_gray_images_(images, labels):
    out_images, out_labels = [], []
    for img, label in zip(images, labels):
        img = np.array(img)
        if img.ndim == 3:
            out_images.append(img)
            out_labels.append(label)

    return np.stack(out_images), np.stack(out_labels)