import os

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image



# DATASET LOCATION

base_folder = "/content/dataset"

folders = ["Dry", "Wet", "Recyclable", "ewaste"]


# CREATE IMAGE PATHS AND LABELS

def labeling():
    img_path = []
    label = []

    for folder in folders:
        idx = folders.index(folder)

        folder_path = os.path.join(base_folder, folder)

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                image_path = os.path.join(root, file)

                img_path.append(image_path)
                label.append(idx)

    return img_path, label


# IMAGE TRANSFORMS

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# DATASET CLASS

class WasteDataset(Dataset):

    def __init__(self, data, label, transform):
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        path = self.data[index]

        pil_img = Image.open(path)
        rgb_img = pil_img.convert("RGB")

        return self.transform(rgb_img), self.label[index]


# CREATE DATALOADERS

def create_dataloaders(img_path, label, batch_size=32):

    # 80% train, 20% temporary
    img_path_train, img_path_temp, label_train, label_temp = train_test_split(
        img_path,
        label,
        train_size=0.8,
        stratify=label,
        random_state=42
    )

    # Split remaining 20% into 10% validation and 10% test
    img_path_valid, img_path_test, label_valid, label_test = train_test_split(
        img_path_temp,
        label_temp,
        train_size=0.5,
        stratify=label_temp,
        random_state=42
    )

    # Create datasets
    train_dataset = WasteDataset(
        img_path_train,
        label_train,
        train_transform
    )

    valid_dataset = WasteDataset(
        img_path_valid,
        label_valid,
        val_transform
    )

    test_dataset = WasteDataset(
        img_path_test,
        label_test,
        test_transform
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_dataloader,
        valid_dataloader,
        test_dataloader
    )