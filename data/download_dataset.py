"""
Download and prepare datasets for training.
Supports: COCO, Pascal VOC, Open Images
"""

import os
import argparse
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm


class DatasetDownloader:
    \"\"\"Handle dataset downloads and preprocessing\"\"\"
    
    def __init__(self, output_dir: str = "./data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_coco(self, year: int = 2017, split: str = "train"):
        \"\"\"
        Download COCO dataset
        
        Args:
            year: Dataset year (2014, 2017)
            split: 'train', 'val', or 'test'
        \"\"\"
        base_url = f"http://images.cocodataset.org/zips"
        
        if split == "train":
            filename = f"train{year}.zip"
        elif split == "val":
            filename = f"val{year}.zip"
        else:
            filename = f"test{year}.zip"
        
        url = f"{base_url}/{filename}"
        output_path = self.output_dir / "coco" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading COCO {split} {year}...")
        self._download_file(url, str(output_path))
        
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_path.parent)
        
        print(f"COCO {split} {year} downloaded successfully!")
    
    def download_pascal_voc(self, year: int = 2012):
        \"\"\"Download Pascal VOC dataset\"\"\"
        url = f"http://host.robots.ox.ac.uk/pascal/VOC/voc{year}/VOCtrainval_{year}.tar"
        filename = f"VOCtrainval_{year}.tar"
        output_path = self.output_dir / "pascal_voc" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading Pascal VOC {year}...")
        self._download_file(url, str(output_path))
        
        print(f"Extracting {filename}...")
        with tarfile.open(output_path, 'r') as tar_ref:
            tar_ref.extractall(output_path.parent)
        
        print(f"Pascal VOC {year} downloaded successfully!")
    
    def create_dataset_split(self, dataset_path: str, train_ratio: float = 0.7, 
                            val_ratio: float = 0.15, test_ratio: float = 0.15):
        \"\"\"
        Split dataset into train/val/test sets
        
        Args:
            dataset_path: Path to dataset
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        \"\"\"
        import random
        import shutil
        
        dataset_path = Path(dataset_path)
        
        # Create split directories
        splits = {
            'train': self.output_dir / 'splits' / 'train',
            'val': self.output_dir / 'splits' / 'val',
            'test': self.output_dir / 'splits' / 'test'
        }
        
        for split_dir in splits.values():
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png'))
        random.shuffle(image_files)
        
        # Calculate split indices
        total = len(image_files)
        train_idx = int(total * train_ratio)
        val_idx = int(total * (train_ratio + val_ratio))
        
        # Split files
        train_files = image_files[:train_idx]
        val_files = image_files[train_idx:val_idx]
        test_files = image_files[val_idx:]
        
        # Copy files
        for files, split in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
            for file in tqdm(files, desc=f"Copying {split} files"):
                shutil.copy2(file, splits[split] / file.name)
        
        print(f"Dataset split created:")
        print(f"  Train: {len(train_files)} images")
        print(f"  Val: {len(val_files)} images")
        print(f"  Test: {len(test_files)} images")
    
    @staticmethod
    def _download_file(url: str, output_path: str):
        \"\"\"Download file with progress bar\"\"\"
        try:
            urllib.request.urlretrieve(url, output_path, reporthook=_download_progress)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            raise


def _download_progress(block_num, block_size, total_size):
    \"\"\"Show download progress\"\"\"
    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100)
    print(f"\rDownload progress: {percent:.1f}%", end='')


def main():
    parser = argparse.ArgumentParser(description='Download object detection datasets')
    parser.add_argument('--dataset', choices=['coco', 'pascal_voc', 'open_images'],
                       default='coco', help='Dataset to download')
    parser.add_argument('--year', type=int, default=2017, help='Dataset year')
    parser.add_argument('--split', choices=['train', 'val', 'test'], 
                       default='train', help='Dataset split to download')
    parser.add_argument('--output', default='./data', help='Output directory')
    parser.add_argument('--create-splits', action='store_true',
                       help='Create train/val/test splits')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.output)
    
    if args.dataset == 'coco':
        downloader.download_coco(year=args.year, split=args.split)
    elif args.dataset == 'pascal_voc':
        downloader.download_pascal_voc(year=args.year)
    
    if args.create_splits:
        dataset_path = Path(args.output) / args.dataset
        downloader.create_dataset_split(str(dataset_path))


if __name__ == '__main__':
    main()
