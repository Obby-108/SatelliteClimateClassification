import torch
from torch.utils.data import IterableDataset

class KoppenDataset(IterableDataset):
    def __init__(self, tf_dataset, transform=None):
        self.tf_dataset = tf_dataset
        self.transform = transform

    def __iter__(self):
        # We handle the batch from TF directly for speed
        for images, labels in self.tf_dataset.as_numpy_iterator():
            # Images shape: (Batch, 129, 129, 12)
            # Permute to PyTorch format: (Batch, 12, 129, 129)
            imgs_pt = torch.from_numpy(images.copy()).float().permute(0, 3, 1, 2)
            labels_pt = torch.from_numpy(labels - 1).long()

            if self.transform:
                # Apply transform to the whole batch at once (Vectorized)
                imgs_pt = self.transform(imgs_pt)

            yield imgs_pt, labels_pt
