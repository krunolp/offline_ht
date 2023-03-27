import torch
import torch.nn as nn
import torch.nn.functional as F


class SqrtNN(nn.Module):

    def __init__(self, epsilon: float = 1.e-3):
        super(SqrtNN, self).__init__()

        assert epsilon >= 0., epsilon
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x.clamp(self.epsilon, None))


class PowerNN(nn.Module):

    def __init__(self, power: float = 2):
        super(PowerNN, self).__init__()

        assert power >= 0., power
        self.power = power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(0., None).pow(self.power)


class FC_MNIST(nn.Module):

    def __init__(self, input_dim=28*28, width=50, depth=3, num_classes=10):
        super(FC_MNIST, self).__init__()
        self.input_dim = input_dim
        self.width = width
        self.depth = depth
        self.num_classes = num_classes

        layers = self.get_layers()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=False),
            nn.ReLU(inplace=True),
            *layers,
            nn.Linear(self.width, self.num_classes, bias=False),
        )

    def get_layers(self):
        layers = []
        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            layers.append(nn.ReLU(inplace=True))
        return layers

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x


class BHP_FCNN(nn.Module):

    def __init__(self, depth: int = 5, width: int = 50, input_dim: int = 13):
        super(BHP_FCNN, self).__init__()
        self.input_dim = input_dim
        self.width = width
        self.depth = depth

        layers = self.get_layers()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.width, bias=True),
            nn.ReLU(inplace=True),
            *layers,
            nn.Linear(self.width, 1, bias=True),
        )

    def get_layers(self):
        layers = []
        for _ in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            layers.append(nn.ReLU())
        return layers

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
        return x


class AttentionFCNN(nn.Module):

    def __init__(self, depth: int = 5, width: int = 100,
                 embedding_size: int = 20, heads: int = 4, input_dim: int = 13):
        super(AttentionFCNN, self).__init__()

        self.input_dim = input_dim
        self.embedding_size = embedding_size
        self.heads = heads
        self.depth = depth
        self.width = width

        assert isinstance(width, int), width
        assert isinstance(heads, int), heads
        assert embedding_size % heads == 0, f"heads should divide embedding_size, but got {heads} and {embedding_size}"
        assert width % embedding_size == 0, f"embedding_size should divide width, but got {embedding_size} and {width}"

        self.first_linear = nn.Linear(self.input_dim, self.width, bias=True)
        self.first_activation = nn.ReLU(inplace=True)

        self.mha_layer = nn.MultiheadAttention(embed_dim=embedding_size,
                                               num_heads=heads,
                                               batch_first=True)

        layers = self.get_layers()
        self.fc = nn.Sequential(
            nn.Linear(self.width, self.width, bias=True),
            nn.ReLU(inplace=True),
            *layers,
            nn.Linear(self.width, 1, bias=True),
        )

    def get_layers(self):
        layers = []
        for _ in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            layers.append(nn.ReLU())
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert x.ndim == 2, x.shape
        assert x.shape[1] == self.input_dim, (x.shape[0], self.input_dim)
        bs = x.shape[0]

        x = self.first_linear(x)
        x = self.first_activation(x)
        x = x.reshape(bs, self.width // self.embedding_size, self.embedding_size)

        x = self.mha_layer(x, x, x, need_weights=False)[0]
        assert x.ndim == 3, x.shape
        x = x.reshape(bs, self.width)

        x = self.fc(x)

        return x


def fc_bhp(**kwargs):
    return BHP_FCNN(**kwargs)


def fc_mnist(**kwargs):
    return FC_MNIST(**kwargs)


class FC_CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3072, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 64)
        self.linear4 = nn.Linear(64, 64)
        self.linear5 = nn.Linear(64, 10)

    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)
        return out


def fc_cifar(**kwargs):
    return FC_CIFAR(**kwargs)


class FC_CIFAR2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3072, 1024)
        self.linear2 = nn.Linear(1024, 64)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, xb):
        out = xb.view(xb.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        return out


def fc_cifar2(**kwargs):
    return FC_CIFAR2(**kwargs)
