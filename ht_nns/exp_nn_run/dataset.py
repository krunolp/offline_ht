import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


class DataOptions:

    def __init__(self, dataset, path, bs_train, bs_eval, resize=None):
        self.dataset = dataset
        self.path = path
        self.batch_size_train = bs_train
        self.batch_size_eval = bs_eval
        self.resize = resize


def get_data_(dataset, path, bs_train, bs_eval, resize=None):
    return get_data(DataOptions(
        dataset,
        path,
        bs_train,
        bs_eval,
        resize
    ))


def get_data(args: DataOptions):
    # mean/std stats
    if args.dataset == 'cifar10':
        data_class = 'CIFAR10'
        num_classes = 10
        stats = {
            'mean': [0.491, 0.482, 0.447],
            'std': [0.247, 0.243, 0.262]
        }
    elif args.dataset == 'cifar100':
        data_class = 'CIFAR100'
        num_classes = 100
        stats = {
            'mean': [0.5071, 0.4867, 0.4408],
            'std': [0.2675, 0.2565, 0.2761]
        }
    elif args.dataset == 'mnist':
        data_class = 'MNIST'
        num_classes = 10
        stats = {
            'mean': [0.1307],
            'std': [0.3081]
        }
    else:
        raise ValueError("unknown dataset")

    # input transformation w/o preprocessing for now

    trans = [
        transforms.ToTensor(),
        lambda t: t.type(torch.get_default_dtype()),
        transforms.Normalize(**stats)
    ]

    if args.dataset == "mnist" and args.resize is not None:
        trans = [
            transforms.ToTensor(),
            lambda t: t.type(torch.get_default_dtype()),
            transforms.Normalize(**stats),
            transforms.Resize(args.resize)
        ]

    # get train and test data with the same normalization
    tr_data = getattr(datasets, data_class)(
        root=args.path,
        train=True,
        download=True,
        transform=transforms.Compose(trans)
    )

    te_data = getattr(datasets, data_class)(
        root=args.path,
        train=False,
        download=True,
        transform=transforms.Compose(trans)
    )

    # get tr_loader for train/eval and te_loader for eval
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_train,
        shuffle=True,
    )

    train_loader_eval = torch.utils.data.DataLoader(
        dataset=tr_data,
        batch_size=args.batch_size_eval,
        shuffle=False,
    )

    test_loader_eval = torch.utils.data.DataLoader(
        dataset=te_data,
        batch_size=args.batch_size_eval,
        shuffle=False,
    )

    return train_loader, test_loader_eval, train_loader_eval, num_classes
