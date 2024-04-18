from typing import Any
import numpy as np

import ignite.distributed as idist
import torchvision
# import torchvision.transforms as T
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

def setup_data(config: Any, is_test = False, few_shot_num = None):
    """Download datasets and create dataloaders

    Parameters
    ----------
    config: needs to contain `data_path`, `train_batch_size`, `eval_batch_size`, and `num_workers`
    """
    local_rank = idist.get_local_rank()

    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    train_transform =  transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    test_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

    if local_rank > 0:
    # Ensure that only rank 0 download the dataset
        idist.barrier()

    if config.dataset.name == "cifar10":

        if is_test:
            raise NotImplementedError("Cifar10 test not implemented.")

        train_dataset = torchvision.datasets.CIFAR10(
            root=config.data_path,
            train=True,
            download=False,
            transform=train_transform,
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root=config.data_path,
            train=False,
            download=False,
            transform=test_transform,
        )
    else:
        name = config.dataset.name
        name = name.replace("-", "")
        print(name)

        import importlib
        dataset_module = importlib.import_module(f"dataloader.{name}")
        namespace = vars(dataset_module)
        public = (name for name in namespace if name[:1] != "_")

        
        matches = [n for n in public if name.lower() == n.lower()]
        print(matches)
        assert len(matches) == 1
        dataset_cls = getattr(dataset_module, matches[0])
        

        train_dataset = dataset_cls(
            config.dataset.root, config.dataset.domain, transform=train_transform
        )
        val_dataset = dataset_cls(
            config.dataset.root, config.dataset.domain, transform=test_transform
        )

        assert len(train_dataset) == len(val_dataset)

        if is_test: # few-shot on target domain
            if few_shot_num is None:
                raise ValueError("few_shot_num should be specified for test dataset.")

            # cnt = [ [] for _ in range(val_dataset.num_classes) ] # CAN'T DO THIS, because task has been divied
            cnt = [ [] for _ in range(config.dataset.num_classes) ]

            indecies = np.random.permutation(len(val_dataset.targets))
            # for i, v in enumerate(val_dataset.targets):
            for i in indecies:
                v = val_dataset.targets[i]
                if len(cnt[v]) < few_shot_num:
                    cnt[v].append(i)
            for i in cnt:
                assert len(i) == few_shot_num
            
            # turn cnt into numpy array and flatten it
            train_indices = np.array(cnt).flatten()
            val_indices = np.array([i for i in range(len(val_dataset)) if i not in train_indices])

            np.random.shuffle(val_indices) # np.random.randint() or sample
            if config.val_sample_num is not None:
                val_indices = val_indices[:config.val_sample_num]
 
            train_dataset = Subset(train_dataset, train_indices)
            val_dataset = Subset(val_dataset, val_indices)
        else:
            # split the dataset with indices
            indices = np.random.permutation(len(train_dataset))
            num_train = int(len(train_dataset) * config.data.train_ratio)
            train_dataset = Subset(train_dataset, indices[:num_train])
            val_dataset = Subset(val_dataset, indices[num_train:])

    if local_rank == 0:
        # Ensure that only rank 0 download the dataset
        idist.barrier()

    # dataloader_train = idist.auto_dataloader(
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        # shuffle=True,
        num_workers=config.num_workers,
    )
    dataloader_eval = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return dataloader_train, dataloader_eval