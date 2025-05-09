import os
import torch
import torchvision
import torch.nn.functional as F


def data_loader(
    rank,
    world_size,
    train=True,
    batch_size=256,
    dataset_name="CIFAR10",
):
    """
    Create a dataloader for CIFAR10 and ImageNet train and test datasets and returns it.

    Parameters:
        - train (bool): whether to return the train or test set.
        - batch_size (int): batch_size to use for the dataloader.
        - dataset_name (str): supports 'CIFAR10' and 'ImageNet'.

    Returns:
        - dataloader (torch.utils.DataLoader): a dataloader.
    """
    # sanity check
    if dataset_name not in ["CIFAR10", "ImageNet"]:
        raise ValueError("We only support 'CIFAR10' and 'ImageNet' datasets.")
    if dataset_name == "CIFAR10":
        path_data = '[mydatapath]' + "/cifar10/"
    else:
        path_data = '[mydatapath]' + "/imagenet/"
    if train:
        dataset = train_data(path_data, dataset_name)
    else:
        dataset = test_data(path_data, dataset_name)
    n_batch_per_epoch = int(len(dataset) / batch_size)

    # creates the dataloader and returns it
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
    )

    return data_loader, n_batch_per_epoch


def train_data(path_data, dataset_name="CIFAR10"):
    """
    Create the training transforms and loads the training dataset for CIFAR10 and ImageNet.

     Parameters:
        - path_data (str): path to the root directory for the dataset.
        - dataset_name (str): supports 'CIFAR10' and 'ImageNet'.

    Returns:
        - dataset_train (torch.utils.Dataset): the training dataset.
    """
    # in the case of the CIFAR10 dataset
    if dataset_name == "CIFAR10":
        # loads CIFAR10 dataset and the corresponding training transforms
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        # CIFAR10 dataset
        dataset_train = torchvision.datasets.CIFAR10(
            root=path_data, train=True, download=False, transform=train_transforms
        )
    # else, we load the ImageNet transforms and dataset
    elif dataset_name == "ImageNet":
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # Imagenet dataset
        dataset_train = torchvision.datasets.ImageNet(
            root=path_data, split="train", transform=train_transforms
        )
    else:
        raise NotImplementedError()

    return dataset_train


def test_data(path_data, dataset_name="CIFAR10"):
    """
    Create the test transforms and loads the test dataset for CIFAR10 and ImageNet.

     Parameters:
        - path_data (str): path to the root directory for the dataset.
        - dataset_name (str): supports 'CIFAR10' and 'ImageNet'.

    Returns:
        - dataset_test (torch.utils.Dataset): the test dataset.
    """
    # in the case of the CIFAR10 dataset
    if dataset_name == "CIFAR10":
        # loads CIFAR10 dataset and the corresponding test transforms
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        # CIFAR10 dataset
        dataset_test = torchvision.datasets.CIFAR10(
            root=path_data, train=False, download=True, transform=test_transforms
        )
    # else, we load the ImageNet transforms and dataset
    elif dataset_name == "ImageNet":
        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # Imagenet dataset
        dataset_test = torchvision.datasets.ImageNet(
            root=path_data, split="val", transform=test_transforms
        )
    else:
        raise NotImplementedError()

    return dataset_test


def evaluate(
    model,
    test_loader,
    criterion,
    local_rank=0,
    print_message=True,
    dataset_name="CIFAR10",
):
    """
    Evaluate the model on the test data, and returns its accuracy.

    Parameters:
        - model (nn.Module): model to evaluate (loaded on a GPU device)
        - test_loader (torch.utils.DataLoader): the test dataloader.
        - criterion (nn.Module): the criterion used to optimize model.
        - local_rank (int): the id of the GPU device the model is loaded on.
        - print_message (bool): whether or not to print the statistics.
        - dataset_name (str): whether or not = 'CIFAR10', to apply logsoftmax after the model outputs.

    Returns:
        - test_loss (float): the average loss over the whole test set.
        - correct (int): the number of correct predictions.
        - dataset_len (int): the number of samples in the dataset.
        - accuracy (float): the accuracy in %.
    """
    # Initiaization
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        # loop over the test set
        for data, target in test_loader:
            # load data to device
            data, target = data.to(local_rank, non_blocking=True), target.to(local_rank, non_blocking=True)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    # average over the number of samples in the dataset
    dataset_len = len(test_loader.dataset)
    dataloader_len = len(test_loader)
    test_loss /= dataloader_len

    if print_message:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, dataset_len, 100.0 * correct / dataset_len
            )
        )
    # put the model back in training mode
    model.train()

    return test_loss, correct, dataset_len, 100.0 * correct / dataset_len