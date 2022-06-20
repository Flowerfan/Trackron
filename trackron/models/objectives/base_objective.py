import torch.nn as nn

class BaseObjective(nn.Module):
    """ Base class for objective. The objective class handles the passing of the data through the network
    and calculation the loss"""
    def __init__(self, loss_func, loss_weight=None):
        """
        args:
            net - The network to train
            loss_funcs - The loss function dict
        """
        super().__init__()
        self.loss_func = loss_func
        self.loss_weight = loss_weight
        

    def forward(self, data, results):
        """ Called in each training iteration. Should pass in input data through the network, calculate the loss, and
        return the training stats for the input data
        args:
            data - A TensorDict containing all the necessary data blocks.
            results - A TensorDict containing all predicted results

        returns:
            loss    - loss for the input data
            stats   - a dict containing detailed losses
        """
        raise NotImplementedError
