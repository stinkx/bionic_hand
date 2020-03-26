import torch
import torch.nn as nn
import Parameter


def weights_init(m):  # evaluated  # TODO: do snother setting for bias zero or not
    if isinstance(m, nn.Conv1d) and Parameter.initializer == 'Xavier_uniform':
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear) and Parameter.initializer == 'Xavier_uniform':
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

    if isinstance(m, nn.Conv1d) and Parameter.initializer == 'Xavier_normal':
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear) and Parameter.initializer == 'Xavier_normal':
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

    if isinstance(m, nn.Conv1d) and Parameter.initializer == 'Kaiming_uniform':
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear) and Parameter.initializer == 'Kaiming_uniform':
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

    if isinstance(m, nn.Conv1d) and Parameter.initializer == 'Kaiming_normal':
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear) and Parameter.initializer == 'Kaiming_normal':
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
