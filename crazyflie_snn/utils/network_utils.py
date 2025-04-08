import numpy as np
import torch

def state_dict_to_weights_array(state_dict):
    """Takes the state dict from a PyTorch model and converts it to a 1-d Numpy array.
    
    Parameters
    ----------
    state_dict : PyTorch model
        state_dict from which weights are extracted

    Returns
    -------
    array
        a numpy array containing all weights from model
    """
    weights_array = []

    for curr_weights in state_dict.values():
        if curr_weights.device != 'cpu':
            curr_weights = curr_weights.to(device='cpu')
        curr_weights = curr_weights.detach().numpy()
        vector = np.reshape(curr_weights, newshape=(curr_weights.size))
        weights_array.extend(vector)

    return np.array(weights_array)


def weights_array_to_state_dict(model, weights_array):
    """Takes a 1-d numpy array and converts it into a PyTorch state dict based on a PyTorch model.

    Parameters
    ----------
    model : PyTorch model
        Model from which structure is used for output dict
    weights_array: numpy array


    Returns
    -------
    dict
        a dict containing all weights in the shape of the model 

    TODO: there is no check implemented to see if weights_array and model correspond
    """

    weights_dict = model.state_dict()

    start = 0
    for key in weights_dict:
        w_matrix = weights_dict[key].detach().cpu().numpy()
        layer_weights_shape = w_matrix.shape
        layer_weights_size = w_matrix.size

        layer_weights_vector = weights_array[start:start + layer_weights_size]
        layer_weights_matrix = np.reshape(layer_weights_vector, newshape=(layer_weights_shape))
        weights_dict[key] = torch.from_numpy(layer_weights_matrix)

        start = start + layer_weights_size

    return weights_dict