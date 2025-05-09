import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    A wrapper around the model, with added functions to ease the communications.
    """

    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, x):
        """
        Perform a forward pass for the classification task.
        """
        output = self.model(x)

        return output
    
    @torch.no_grad()
    def get_weights(self,):
        """
        Wrapper around nn.utils.parameters_to_vector.
        Given a nn.Module, returns a 1D tensor containing all of its parameters.
        """
        learnable_params =  nn.utils.parameters_to_vector(self.model.parameters())
        
        # store the size of the learnable params
        self.size_learnable_params = len(learnable_params)
        bn_params = []
        # iterate over the batch norm "non learnable" params
        for mod in self.model.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                bn_params.append(mod.running_mean.view(-1))
                bn_params.append(mod.running_var.view(-1))
        if len(bn_params) > 0:
            bn_params = torch.cat(bn_params)
            return torch.cat([learnable_params, bn_params])
        else:
            return learnable_params
    

    @torch.no_grad()
    def set_weights(self, weights):
        """
        Wrapper around nn.utils.vector_to_parameters.
        Given a 1D tensor containing a nn.Module parameters,
        loads the parameters into the nn.Module.
        """
        nn.utils.vector_to_parameters(weights[:self.size_learnable_params], self.model.parameters())
        
        # Loads the bn non learnable params
        # Pointer for slicing the vector for each parameter
        pointer = self.size_learnable_params
        # iterate over the batch norm "non learnable" params
        for mod in self.model.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                for param in [mod.running_mean, mod.running_var]:
                    # The length of the parameter
                    num_param = param.numel()
                    # Slice the vector, reshape it, and replace the old data of the parameter
                    param.data = weights[pointer:pointer + num_param].view_as(param).data
                    # Increment the pointer
                    pointer += num_param



def create_model(model_name="resnet18", dataset_name="CIFAR10"):
    """
    Returns the model corresponding to the given name,
    modified (or not) to handle the given dataset.

    Parameters:
        - model_name (str): the name of the model to load.
                            either one of ['resnet18', ]
        - dataset_name (str): the name of the dataset to use.
                              either one of ['CIFAR10', 'ImageNet']
    Returns:
        - net (nn.Module): the neural net to use.
        - criterion (nn.Module): the criterion to use.
    """
    # sanity check
    if dataset_name not in ["CIFAR10", "ImageNet"]:
        raise ValueError("We only support 'CIFAR10' and 'ImageNet' datasets.")
    if model_name not in ["resnet18", "resnet50"]:
        raise ValueError("We only support 'resnet18' and 'resnet50'.")

    # Initialization fit for ImageNet
    modified_conv1 = None
    modified_maxpool = None
    num_classes = 1000
    criterion = nn.CrossEntropyLoss(reduction="mean")

    # adapt to the specificities of CIFAR10
    if dataset_name == "CIFAR10":
        # set the correct number of classes,
        num_classes = 10
        # modify the first conv to handle the image size
        modified_conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        # modify the maxpool layer
        modified_maxpool = nn.Identity()
    # loads the model
    weights = None
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights=weights, num_classes=num_classes)
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(weights=weights, num_classes=num_classes)
    # update the model if need be
    if modified_conv1 is not None:
        model.conv1 = modified_conv1
    if modified_maxpool is not None:
        model.maxpool = modified_maxpool

    return Net(model), criterion


def compute_multiplicative_coef_lr(
    k_step,
    n_step_tot,
    n_epoch_if_1_worker,
    lr,
    world_size,
    batch_size,
    dataset_name,
    return_function=False,
):
    """
    A function returning a function that generate the modified lr schedule for large batch size of Goyal et al. https://arxiv.org/pdf/1706.02677.pdf
    This function is then used in the LambdaLR scheduler of pytorch.

    Parameters:
        - k_step (int): the current value of the iteration counter.
        - n_step_tot (int): the total number of iterations.
        - n_epoch_if_1_worker (int): the total number of epochs.
        - lr (float): the base learning rate.
        - world_size (int): the number of workers.
        - batch_size (int): the batch size per worker.
        - dataset_name (str): the name of the datasets (as scheduler differ for ImageNet and CIFAR10).
        - return_function (bool): whether to return a lambda function, or the value of the multiplicative factor to the base
                                  learning rate at the current step.

    Returns:
        - Either one of the multiplicative factor, or a function to use in LambdaLR, depending on return_function.
    """

    # init of the constants
    n_step_per_epoch = n_step_tot // n_epoch_if_1_worker
    five_epoch_step = 5 * n_step_per_epoch
    milestones = [five_epoch_step]
    multiplicative_factor = 1
    # create the milestones, depending on the dataset
    if dataset_name == "CIFAR10":
        # put float('Inf') be sure the last stage lasts all of the remaining of the training
        milestones += [int(0.5 * n_step_tot), int(0.75 * n_step_tot), float('Inf')]
    else:
        milestones += [
            int(0.3 * n_step_tot),
            int(0.6 * n_step_tot),
            int(0.8 * n_step_tot),
            float('Inf'),
        ]
    # create the linear warm up from base lr to the value of
    # lr x (bs/256) x world_size for the 5 first epochs
    if k_step < five_epoch_step:
        linear_slope = (1 / five_epoch_step) * (batch_size * world_size / 256 - 1)
        multiplicative_factor = 1 + k_step * linear_slope
    else:
        for k in range(len(milestones) - 1):
            if k_step >= milestones[k] and k_step < milestones[k + 1]:
                multiplicative_factor = (0.1**k) * batch_size * world_size / 256
    # if we return a function, we return a mapping from k_step to the multiplicative factor to use
    # setting return_function to False for that.
    if return_function:
        return lambda step: compute_multiplicative_coef_lr(
            step * world_size,  # assumes that all worker work at the same speed
            n_step_tot,
            n_epoch_if_1_worker,
            lr,
            world_size,
            batch_size,
            dataset_name,
            return_function=False,
        )
    # else, we directly return the multiplicative factor
    else:
        return multiplicative_factor


def add_weight_decay(model, weight_decay, skip_list=()):
    """
    Create 2 sets of parameters: ones to which weight decay should be applied to, and the others (batch norm and bias terms).
    Credit to Ross Wightman https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3

    Parameters:
        - model (nn.Module): the neural net to use.
        - weight_decay (float): value of the weight decay.
        - skip_list (list): the list of parameters names to add to the skip_list.

    Returns:
        - params_list (list of dicts): two sets of parameters, one where no wd is used,
                                       the other where wd is applied.
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # in ResNets, len(param.shape) is one only for batch norm weights and bias terms.
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def load_optimizer(
    model,
    lr,
    momentum,
    weight_decay,
    gamma,
    optimizer_name,
    world_size,
    use_linear_scaling,
    n_step_tot,
    n_epoch_if_1_worker,
    batch_size,
    filter_bias_and_bn=True,
    dataset_name="CIFAR10",
):
    """
    Returns the optimizer corresponding to the given name,
    as well as a lr scheduler.

    Parameters:
        - model (nn.Module): the model to which we apply the optimizer.
        - lr (float): the learning rate.
        - momentum (float): the momentum value.
        - weight_decay (float): the weight decay value.
        - gamma (float): the gamma parameter of the StepLR scheduler.
        - optimizer_name (str): the name of the optimizer to load.
                                either one of ['SGD']
        - world_size (int): the number of workers.
        - use_linear_scaling (bool): whether or not to use the LR schedule for large batch sizes.
        - n_step_tot (int): the total number of iterations.
        - n_epoch_if_1_worker (int): the total number of epochs.
        - batch_size (int): the batch size per worker.
        - filter_bias_and_bn (bool): whether or not to create two sets of parameters to avoid applying wd to bn params.
        - datset_name (str): name of the dataset.

    Returns:
        - optimizer (Optimizer): the optimizer to use.
        - lr_scheduler (Scheduler): the initialized lr scheduler.
    """
    # sanity check
    if optimizer_name not in ["SGD"]:
        raise ValueError("We only support optimizers in ['SGD'].")
    # loads the model parameters
    if weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay(model, weight_decay)
        wd = 0.0
    else:
        parameters = model.parameters()
        wd = weight_decay
    # loads the optimizer
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            parameters, lr=lr, momentum=momentum, weight_decay=wd
        )
    # loads the lr scheduler
    if use_linear_scaling:
        # use the scheduler from https://arxiv.org/pdf/1706.02677.pdf
        f = compute_multiplicative_coef_lr(
            0,
            n_step_tot,
            n_epoch_if_1_worker,
            lr,
            world_size,
            batch_size,
            dataset_name,
            return_function=True,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, f)
    else:
        # use a simple step LR
        if dataset_name == "CIFAR10":
            milestones = [int(0.5 * n_step_tot), int(0.75 * n_step_tot)]
        else:
            milestones = [
                int(0.3 * n_step_tot),
                int(0.6 * n_step_tot),
                int(0.8 * n_step_tot),
            ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones, gamma=gamma
        )

    return optimizer, scheduler