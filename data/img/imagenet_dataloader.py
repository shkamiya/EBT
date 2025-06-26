from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import getpass
from torchvision import transforms

#credit: https://huggingface.co/datasets/imagenet-1k'
#NOTE: if you are having issues with this dataloader and perms you need to add your HF token
#see these links - https://discuss.huggingface.co/t/imagenet-1k-is-not-available-in-huggingface-dataset-hub/25040 https://huggingface.co/docs/hub/security-tokens
class ImageNetDataset(Dataset):
    def __init__(self, hparams, split, transform):
        self.hparams = hparams
        self.transform = transform
        current_user = getpass.getuser()
        split = 'validation' if split in ["valid", "val", "validate"] else split
        dataset_dir = self.hparams.dataset_dir if self.hparams.dataset_dir != "" else f"/scratch/{current_user}/imagenet/"
        self.ds = load_dataset("imagenet-1k", cache_dir = dataset_dir, trust_remote_code=True, split=split)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        sample = self.ds[idx]
        image = sample['image']
        image_mode = image.mode
        if image_mode == 'L':
            image = image.convert("RGB")
        elif image_mode == 'RGBA':
            image = image.convert("RGB")
        
        transformed_image = self.transform(image)
        data_dict = {
            'transformed_images' : transformed_image,
            'labels' : sample['label']
        }
        return data_dict

