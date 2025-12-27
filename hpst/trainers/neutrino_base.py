import lightning.pytorch as pl
import torch

# noinspection PyProtectedMember
from torch.utils.data import DataLoader,Subset, Dataset


from hpst.utils.options import Options
from hpst.utils.learning_rate_schedules import get_linear_schedule_with_warmup
from hpst.utils.learning_rate_schedules import get_cosine_with_hard_restarts_schedule_with_warmup, get_cosine_annealing_with_warmup
import torch_geometric.transforms as T


class NeutrinoBase(pl.LightningModule):
    def __init__(self, options: Options, train_perc=None):
        super(NeutrinoBase, self).__init__()

        self.options = options
        self.train_perc = train_perc

        self.training_dataset, self.validation_dataset, self.testing_dataset = self.create_datasets()

        self.mean = 0
        self.std = 1

        self.extra_mean = 0
        self.extra_std = 1

        self.pixel_mean = 0
        self.pixel_std = 1

        # Normalize datasets using training dataset statistics
        if self.options.normalize_features:
            # --- FIX: Subset has no compute_statistics() / pixels ---
            base_train = self.training_dataset
            if isinstance(self.training_dataset, Subset):
                base_train = self.training_dataset.dataset

            (self.mean, self.std,
             self.extra_mean, self.extra_std,
             self.pixel_mean, self.pixel_std) = base_train.compute_statistics()

            self.mean = torch.nn.Parameter(self.mean, requires_grad=False)
            self.std = torch.nn.Parameter(self.std, requires_grad=False)

            self.extra_mean = torch.nn.Parameter(self.extra_mean, requires_grad=False)
            self.extra_std = torch.nn.Parameter(self.extra_std, requires_grad=False)

            # robust check for pixels attribute
            pixels = getattr(base_train, "pixels", None)
            if pixels is not None:
                self.pixel_mean = torch.nn.Parameter(self.pixel_mean, requires_grad=False)
                self.pixel_std = torch.nn.Parameter(self.pixel_std, requires_grad=False)

        self.steps_per_epoch = len(self.training_dataset) // (self.options.batch_size * max(1, self.options.num_gpu))
        self.total_steps = self.steps_per_epoch * self.options.epochs
        self.warmup_steps = int(round(self.steps_per_epoch * self.options.learning_rate_warmup_epochs))

    @property
    def dataset(self):
        return Dataset

    @property
    def dataloader(self):
        return DataLoader

    @property
    def dataloader_options(self):
        return {
            "drop_last": True,
            "batch_size": self.options.batch_size,
            "pin_memory": self.options.num_gpu > 0,
            "num_workers": self.options.num_dataloader_workers
        }

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.options.optimizer, None)
        if optimizer is None:
            print(f"Unable to load desired optimizer: {self.options.optimizer}.")
            print(f"Using pytorch AdamW as a default.")
            optimizer = torch.optim.AdamW

        optimizer = optimizer(self.parameters(), lr=self.options.learning_rate, weight_decay=self.options.l2_penalty)

        if self.options.learning_rate_cycles < 1:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps
            )
        else:
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
                num_cycles=self.options.learning_rate_cycles
            )

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]
#NOvA
    # def create_datasets(self):
    #     transform = T.Compose([
    #         # T.RandomJitter(5),
    #         T.RandomScale((0.99,1.01)),
    #         T.RandomRotate((-1,1),axis=0),
    #         T.RandomRotate((-1,1),axis=1),
    #         # T.RandomShear(0.1),
    #     ])
    #     print("loading training")
    #     if len(self.options.validation_file) > 0:
    #         training_dataset = self.dataset(self.options.training_file, transform=transform)
    #         validation_dataset = self.dataset(self.options.validation_file)
    #     else:
    #         # Compute the training / validation ranges based on the data-split and the limiting percentage.
    #         train_validation_split = self.options.dataset_limit * self.options.train_validation_split
    #         if self.train_perc is None:
    #             train_range = (0.0, train_validation_split)
    #             validation_range = (train_validation_split, self.options.dataset_limit)
    #         else:
    #             train_range = (0.0, train_validation_split*self.train_perc)
    #             validation_range = (train_validation_split, train_validation_split + self.train_perc*(self.options.dataset_limit - train_validation_split))

    #         training_dataset = self.dataset(self.options.training_file, train_range, transform=transform)
    #         validation_dataset = self.dataset(self.options.training_file, validation_range)
        
    #     print("loading testing")
    #     testing_dataset = None
    #     if len(self.options.testing_file) > 0:
    #         test_range = (0.0, 1.0)
    #         testing_dataset = self.dataset(self.options.testing_file, test_range, testing=True)

    #     return training_dataset, validation_dataset, testing_dataset

#JUNO
    def create_datasets(self):
        """
        For JUNO directory dataset:
        - Load full dataset from options.training_file
        - Split into train/val/test by ratios so it can be controlled from .json
        """
        # 确保子类覆盖了 dataset property
        if self.dataset is Dataset:
            raise RuntimeError(
                "You must override the `dataset` property in your trainer (e.g., PointSetTrainer) "
                "to return the concrete dataset class (e.g., JUNOTQPairHitDataset), not the base Dataset."
            )
        
        print("loading JUNO dataset (full)")
        full_dataset = self.dataset(self.options.training_file, options=self.options)

        n_total = len(full_dataset)
        if n_total <= 1:
            raise RuntimeError(f"Dataset too small: n_total={n_total}")

        test_split = float(getattr(self.options, "test_split", 0.0))
        train_val_split = float(getattr(self.options, "train_validation_split", 0.95))
        split_seed = int(getattr(self.options, "split_seed", 12345))

        if not (0.0 <= test_split < 1.0):
            raise ValueError(f"options.test_split must be in [0,1), got {test_split}")
        if not (0.0 < train_val_split < 1.0):
            raise ValueError(f"options.train_validation_split must be in (0,1), got {train_val_split}")

        # If user provides a separate testing_file directory, prefer that as testing dataset.
        testing_dataset = None
        if isinstance(self.options.testing_file, str) and len(self.options.testing_file) > 0:
            print("loading testing (separate directory)")
            testing_dataset = self.dataset(self.options.testing_file, options=self.options)

        # Deterministic permutation
        g = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(n_total, generator=g).tolist()

        # Split indices
        n_test = 0 if testing_dataset is not None else int(round(n_total * test_split))
        n_test = max(0, min(n_test, n_total - 2))  # keep at least 2 samples for train/val

        remaining = n_total - n_test
        n_train = int(round(remaining * train_val_split))
        n_train = max(1, min(n_train, remaining - 1))  # keep at least 1 for val
        n_val = remaining - n_train

        test_idx = perm[:n_test]
        train_idx = perm[n_test:n_test + n_train]
        val_idx = perm[n_test + n_train:]

        print(f"split: total={n_total} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} seed={split_seed}")

        training_dataset = Subset(full_dataset, train_idx)
        validation_dataset = Subset(full_dataset, val_idx)

        if testing_dataset is None and n_test > 0:
            testing_dataset = Subset(full_dataset, test_idx)

        return training_dataset, validation_dataset, testing_dataset

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.training_dataset, shuffle=True, **self.dataloader_options)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.validation_dataset, **self.dataloader_options)

    def test_dataloader(self) -> DataLoader:
        if self.testing_dataset is None:
            raise ValueError("Testing dataset not provided.")
        return self.dataloader(self.testing_dataset, **self.dataloader_options)