import torch
import torch.nn as nn
#import Parameter

# def choose_init(initializer):
#     if initializer == 'Xavier_uniform':
#         def weights_init(m):  # evaluated  # TODO: do another setting for bias zero or not
#             if isinstance(m, nn.Conv1d) and initializer == 'Xavier_uniform':
#                 torch.nn.init.xavier_uniform_(m.weight)
#                 torch.nn.init.zeros_(m.bias)
#             if isinstance(m, nn.Linear) and initializer == 'Xavier_uniform':
#                 torch.nn.init.xavier_uniform_(m.weight)
#                 torch.nn.init.zeros_(m.bias)
#
#         return weights_init()


def weights_init(m):  # evaluated  # TODO: do another setting for bias zero or not
    #initializer = 'Xavier_uniform'
    if isinstance(m, nn.Conv1d) and initializer == 'Xavier_uniform':
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear) and initializer == 'Xavier_uniform':
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

    if isinstance(m, nn.Conv1d) and initializer == 'Xavier_normal':
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear) and initializer == 'Xavier_normal':
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

    if isinstance(m, nn.Conv1d) and initializer == 'Kaiming_uniform':
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear) and initializer == 'Kaiming_uniform':
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

    if isinstance(m, nn.Conv1d) and initializer == 'Kaiming_normal':
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear) and initializer == 'Kaiming_normal':
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
