import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------------------------------------------------
# Filter CSV rows to keep only existing image files
# ---------------------------------------------------

def filter_csv_by_existing_files(csv_path, base_data_dir, output_path):
    df = pd.read_csv(csv_path)

    def file_exists(rel_path):
        full_path = os.path.join(base_data_dir, rel_path.strip())
        return os.path.isfile(full_path)

    filtered_df = df[df["Path"].apply(file_exists)]
    filtered_df.to_csv(output_path, index=False)
    return filtered_df


# ---------------------------------------------------
# Traffic Sign Dataset with optional ROI cropping
# ---------------------------------------------------

class TrafficSignDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None, use_roi=False):
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        self.use_roi = use_roi

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.data_dir, row['Path'])
        image = Image.open(img_path).convert("RGB")

        if self.use_roi:
            x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
            image = image.crop((x1, y1, x2, y2))

        if self.transform:
            image = self.transform(image)

        label = int(row["ClassId"])
        return image, label


# ---------------------------------------------------
# Data transforms (augmentation + test transforms)
# ---------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])


# ---------------------------------------------------
# Loaders for train/val/test
# ---------------------------------------------------

def get_loaders(train_csv, val_csv, test_csv, data_dir, batch_size=64, use_roi=True):
    train_dataset = TrafficSignDataset(train_csv, data_dir, train_transform, use_roi)
    val_dataset   = TrafficSignDataset(val_csv, data_dir, test_transform, use_roi)
    test_dataset  = TrafficSignDataset(test_csv, data_dir, test_transform, use_roi)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
