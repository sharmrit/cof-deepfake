import torch
from torch.utils.data import DataLoader, Sampler
import numpy as np

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.classes = np.unique(self.labels)
        self.n_classes = len(self.classes)
        self.n_per_class = batch_size // self.n_classes
        self.class_indices = {c: np.where(self.labels == c)[0] for c in self.classes}
        self.n_batches = min(len(idx) // self.n_per_class for idx in self.class_indices.values())

    def __iter__(self):
        class_iters = {c: iter(np.random.permutation(idx)) for c, idx in self.class_indices.items()}
        for _ in range(self.n_batches):
            batch = []
            for c in self.classes:
                for _ in range(self.n_per_class):
                    batch.append(next(class_iters[c]))
            np.random.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.n_batches * self.batch_size

def create_data_loaders(train_dataset, val_dataset, test_dataset,
                        batch_size=64, num_workers=4, balanced=True):
    if balanced and hasattr(train_dataset, "labels"):
        sampler = BalancedBatchSampler(train_dataset.labels, batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
