import torch


# implement learning rate decay too!
def get_optimizer(optimizer_name, parameters, learning_rate, weight_decay, momentum):
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
    else:
        optimizer = False
        print('Invalid Optimizer!')

    return optimizer
