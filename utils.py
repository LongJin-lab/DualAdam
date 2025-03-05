import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os, random
import numpy as np
from torch import optim

from torch.utils.data import DataLoader

from DualAdam import DualAdam
from model import resnet, vgg, vit

'''数据集准备'''


def load_data(dataset_name):
    trainloader = None
    testloader = None
    if dataset_name == "cifar10":
        # data preprocessing
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # CIFAR-10 training set
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        # CIFAR-10 test set
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    elif dataset_name == "cifar100":
        # data preprocessing
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # CIFAR-100 training set
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

        # CIFAR-100 test set
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    elif dataset_name == "tiny_imagenet":
        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        transform_train = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(),
             normalize, ])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize, ])
        trainset = datasets.ImageFolder(root=os.path.join('./data/tiny_imagenet/tiny-imagenet-200', 'train'), transform=transform_train)
        testset = datasets.ImageFolder(root=os.path.join('./data/tiny_imagenet/tiny-imagenet-200', 'val'), transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, pin_memory=True)
    elif dataset_name == "imagenet1k":
        # data preprocessing
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        # ImageNet training set
        train_dataset = datasets.ImageFolder(
            root='/data/imagenet/train',  # 替换为ImageNet训练集的路径
            transform=train_transforms
        )

        # ImageNet test set
        test_dataset = datasets.ImageFolder(
            root='/data/imagenet/val',  # 替换为ImageNet验证集的路径
            transform=test_transforms
        )

        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
        )

        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
        )
    return trainloader, testloader


def train(net, trainloader, optimizer, criterion, device="cuda"):
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    net.train()
    running_loss = 0.0
    for data, label in trainloader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        with autocast:
            output = net(data)
            loss = criterion(output, label)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()
        running_loss += loss.item()
    return running_loss / len(trainloader)


def test(net, testloader, device="cuda"):
    net.eval()
    correct = 0
    total = 0
    autocast = torch.cuda.amp.autocast
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            with autocast():
                output = net(data)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
    return 100 * correct / total


def average_accuracy(accuracys, epoch_num):
    total = 0
    for accuracy in accuracys[-epoch_num:]:
        total += accuracy
    return total / (epoch_num + 0.0)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


def select_model(model_name, device):
    if model_name == "resnet18":
        return resnet.ResNet18(num_classes=1000).to(device)
    elif model_name == "resnet50":
        return resnet.ResNet50(num_classes=1000).to(device)
    elif model_name == "vgg16":
        return vgg.VGG(vgg_name="VGG16", num_classes=1000).to(device)
    elif model_name == "vits4":
        return vit.VisionTransformer(img_size=32,
                                     patch_size=4,
                                     embed_dim=384,
                                     depth=8,
                                     num_heads=8,
                                     representation_size=None,
                                     num_classes=1000).to(device)
    elif model_name == "vits8":
        return vit.VisionTransformer(img_size=64,
                                     patch_size=8,
                                     embed_dim=384,
                                     depth=8,
                                     num_heads=8,
                                     representation_size=None,
                                     num_classes=1000).to(device)
    elif model_name == "vits16":
        return vit.VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=384,
            depth=8,
            num_heads=8,
            representation_size=None,
            num_classes=1000
        ).to(device)


def select_optimizer(optimizer_name, model, lr):
    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=lr, weight_decay=0)
    elif optimizer_name == "DualAdam":
        return DualAdam(params=model.parameters(), lr=lr, beta1=0.9, beta2=0.999, epsilon=1e-8,
                        switch_rate=8e-5, weight_decay=0)
    elif optimizer_name == "AdamW":
        return optim.AdamW(model.parameters(), betas=(0.9, 0.999), eps=1e-8, lr=lr, weight_decay=1e-2)
    elif optimizer_name == "RAdam":
        return optim.RAdam(model.parameters(), lr=lr, weight_decay=1e-2, decoupled_weight_decay=True)
    elif optimizer_name == "NAdam":
        return optim.NAdam(model.parameters(), lr=lr, weight_decay=0, decoupled_weight_decay=True, momentum_decay=4e-3)



