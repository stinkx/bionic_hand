import torch


# implement learning rate decay too!
def get_optimizer(optimizer_name, parameters, learning_rate, weight_decay, momentum):
    if type(optimizer_name) is not str or optimizer_name not in ["SGD", "Adam", "Adagrad", "Adadelta", "RMSprop"]:
        raise ValueError("Optimizer name must be a string and match one of the implemented optimizer.")
    if type(learning_rate) is not float or learning_rate < 0 or learning_rate >= 1:
        raise ValueError("Learning rate cannot be negative and should not be larger than one.")
    if type(weight_decay) is not float or weight_decay < 0 or weight_decay >= 1:
        raise ValueError("Weight decay cannot be negative and should not be larger than one.")
    if type(momentum) is not float or momentum < 0 or momentum >= 1:
        raise ValueError("Momentum cannot be negative and should not be larger than one.")

    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'Adadelta':
        optimizer = torch.optim.Adadelta(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    return optimizer
