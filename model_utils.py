import os
import torch
import torchaudio
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# Audio Dataset
# =============================
class AudioSpectrogramDataset(Dataset):
    def __init__(self, df, sr=16000, n_mels=64, max_len=256):
        self.df = df.reset_index(drop=True)
        self.sr = sr
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=400, hop_length=160, n_mels=n_mels
        )
        self.db = torchaudio.transforms.AmplitudeToDB()
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path, label = self.df.loc[idx, "filepath"], self.df.loc[idx, "label"]
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        spec = self.mel(wav)
        spec = self.db(spec)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        if spec.shape[-1] < self.max_len:
            pad = self.max_len - spec.shape[-1]
            spec = torch.nn.functional.pad(spec, (0, pad))
        else:
            spec = spec[:, :, :self.max_len]

        return spec, torch.tensor(label).long()

# =============================
# X-ray Dataset
# =============================
class XrayDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        img = torchvision.io.read_image(img_path).float() / 255.0
        if img.shape[0] == 1:
            img = img.repeat(3,1,1)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row['label']).long()
        return img, label

# =============================
# Model Builders
# =============================
def build_resnet_audio(n_classes=2):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)

def build_resnet_xray(n_classes=2):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)
