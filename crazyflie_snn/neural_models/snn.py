import torch

from spiking.core.torch.model import BaseModel
from spiking.core.torch.layer import LinearCubaLif, LinearIdentity, LinearCubaSoftLif, LinearRnnCubaLif, LinearRnnCubaSoftLif, LinearLi


class BaseSNN(BaseModel):
    """
    Base class for SNNs.
    """

    def forward(self, input):
        '''
        Forward pass through the network.
        Simplest implementation is to just pass the input through each layer in turn.
        This does not allow for any skip connections or other more complex architectures.
        TODO: Implement logging between layers?.
        '''
        for layer in self.layers:
            input = layer(input)
            # if len(layer.state) == 3:
            #     # print(f"in, i, v, s, th: {input[0, 0]},{layer.state[2][0][0, 0]}, {layer.state[2][1][0, 0]}, {layer.state[2][2][0, 0]}, {layer.neuron.thresh[0]}")
            #     print(layer.state[2][2][0, :])
            # if len(layer.state) == 2:
            #     if len(layer.state[1]) == 3:
            #         # print(f"in, i, v, s, th: {input[0, 0]}, {layer.state[1][0][0, 0]}, {layer.state[1][1][0, 0]}, {layer.state[1][2][0, 0]}, {layer.neuron.thresh[0]}")
            #         print(layer.state[1][2][0, :])
        return input
    
    def forward_sequence(self, input):
        output = []
        for i in range(input.size()[0]):
            output.append(self(input[i]))
        return torch.stack(output)
    
    def reset(self, device='cpu'):
        '''
        Reset the states of the network.
        '''
        for layer in self.layers:
            layer.reset()
        
class SNN(BaseSNN):
    """
    Two layer SNN.
    """

    def __init__(self, l1, l2, p_out):
        super().__init__()
        self.l1 = LinearCubaLif(**l1)
        self.l2 = LinearCubaLif(**l2)
        self.p_out = LinearIdentity(**p_out)
        self.layers = torch.nn.ModuleList([self.l1, self.l2, self.p_out])


class OneLayerSNN(BaseSNN):
    """
    One layer SNN.
    """
    def __init__(self, l1, p_out):
        super().__init__()
        self.l1 = LinearCubaLif(**l1)
        self.p_out = LinearIdentity(**p_out)
        self.layers = torch.nn.ModuleList([self.l1, self.p_out])


class IntegrateSNN(BaseSNN):
    '''
    Class that is constructed to perform integration
    '''
    def __init__(self, l1, p_out):
        super().__init__()
        self.l1 = LinearCubaSoftLif(**l1)
        self.p_out = LinearIdentity(**p_out)
        self.layers = torch.nn.ModuleList([self.l1, self.p_out])


class RSNN(BaseSNN):
    """
    Two layer SNN with the second layer spiking.
    """

    def __init__(self, l1, l2, p_out):
        super().__init__()
        self.l1 = LinearCubaLif(**l1)
        self.l2 = LinearRnnCubaLif(**l2)
        self.p_out = LinearIdentity(**p_out)
        self.layers = torch.nn.ModuleList([self.l1, self.l2, self.p_out])
    

class OneLayerRSNN(BaseSNN):
    """
    One layer RSNN.
    """

    def __init__(self, l1, p_out):
        super().__init__()
        self.l1 = LinearRnnCubaLif(**l1)
        self.p_out = LinearIdentity(**p_out)
        # self.p_out = LinearLi(**p_out)
        self.layers = torch.nn.ModuleList([self.l1, self.p_out])

    def set_weights(self, n_fixed):
        # new_output_weights = self.p_out.synapse.weight.clone()
        # new_output_weights = new_output_weights * 0.01
        # self.p_out.synapse.weight = torch.nn.Parameter(new_output_weights)

        # new_leak_i = self.l1.neuron.leak_i.clone()
        # new_leak_i[:n_fixed] = torch.inf
        # self.l1.neuron.leak_i = torch.nn.Parameter(new_leak_i)
        # new_leak_v = self.l1.neuron.leak_v.clone()  
        # new_leak_v[:n_fixed] = torch.inf
        # self.l1.neuron.leak_v = torch.nn.Parameter(new_leak_v)
        # self.neuron_mask = torch.ones_like(self.l1.neuron.leak_i)
        # self.neuron_mask[:n_fixed] = 0
        pass

    def mask_grads(self):
        # self.l1.neuron.leak_i.grad *= self.neuron_mask.to(self.l1.neuron.leak_i.device)
        # self.l1.neuron.leak_v.grad *= self.neuron_mask.to(self.l1.neuron.leak_v.device)
        self.l1.synapse_rec.weight.grad *= self.neuron_mask_rec.to(self.l1.synapse_rec.weight.device)
    
    def detach_hidden(self):
        for i in range(len(self.l1.state)):
            self.l1.state[i] = [state.detach() for state in self.l1.state[i]]
            # self.l1.state[i] = [torch.zeros_like(state) for state in self.l1.state[i]]
        for i in range(len(self.p_out.state)):
            self.p_out.state[i] = [state.detach() for state in self.p_out.state[i]]

class OneLayerSoftRSNN(BaseSNN):
    """
    One layer SNN with soft reset.
    """

    def __init__(self, l1, p_out, combined):
        super().__init__()
        self.l1 = LinearRnnCubaSoftLif(**l1)
        self.p_out = LinearLi(**p_out)
        self.combined = LinearIdentity(**combined)
        self.layers = torch.nn.ModuleList([self.l1, self.p_out, self.combined])

    def set_weights(self, n_fixed, device):
        fix_grads = False

        # set the inputs weights for integration to be smaller
        new_input_weights = self.l1.synapse_ff.weight.clone()
        new_input_weights[:n_fixed, :] = new_input_weights[:n_fixed, :] * 0.1
        self.l1.synapse_ff.weight = torch.nn.Parameter(new_input_weights)

        # set the output weights for integration to be small
        new_output_weights = self.p_out.synapse.weight.clone()
        print(new_output_weights.size())
        new_output_weights[:2, :n_fixed] = new_output_weights[:2, :n_fixed] * 0.05
        new_output_weights[2:, :n_fixed] = 0
        new_output_weights[:2, n_fixed:] = 0        
        self.p_out.synapse.weight = torch.nn.Parameter(new_output_weights)

        # set the output leaks for integration to be infinite and the others zero
        new_leak_i = self.p_out.neuron.leak_i.clone()
        new_leak_i[2:] = 0
        self.p_out.neuron.leak_i = torch.nn.Parameter(new_leak_i, requires_grad=fix_grads)
        new_leak_v = self.p_out.neuron.leak_v.clone()
        new_leak_v[2:] = 0
        self.p_out.neuron.leak_v = torch.nn.Parameter(new_leak_v, requires_grad=fix_grads)

        # Set the leaks for integration to be infinite and create mask
        new_leak_i = self.l1.neuron.leak_i.clone()
        new_leak_i[:n_fixed] = torch.inf
        self.l1.neuron.leak_i = torch.nn.Parameter(new_leak_i)
        new_leak_v = self.l1.neuron.leak_v.clone()
        new_leak_v[:n_fixed] = torch.inf
        self.l1.neuron.leak_v = torch.nn.Parameter(new_leak_v)
        self.neuron_mask = torch.ones_like(self.l1.neuron.leak_i)
        self.neuron_mask[:n_fixed] = 0

        # Set the recurrent weights for integration to be zero and create mask
        new_weights = self.l1.synapse_rec.weight.clone()
        new_weights[:n_fixed, :] = 0
        new_weights[:, :n_fixed] = 0
        self.l1.synapse_rec.weight = torch.nn.Parameter(new_weights)
        self.neuron_mask_rec = torch.ones_like(self.l1.synapse_rec.weight)
        self.neuron_mask_rec[:n_fixed, :] = 0
        self.neuron_mask_rec[:, :n_fixed] = 0

        # set the combination weights for final output
        new_combined_weights = self.combined.synapse.weight.clone()
        # new_combined_weights = torch.Tensor([[1, 0, 1, 0, 0, 0],
        #                                      [0, 1, 0, 1, 0, 0],
        #                                      [0, 0, 0, 0, 1, 1]])
        new_combined_weights = torch.Tensor([[1, 0, 0, 0, 0, 0]])
        self.combined.synapse.weight = torch.nn.Parameter(new_combined_weights, requires_grad=fix_grads)
        self.to(device)
        self.do_mask_grads = True


    def set_weights_from_vectors(self, ff_weights, rec_weights, l1_leak_i, l1_leak_v, l1_threshold, out_weights):
        self.l1.synapse_ff.weight = torch.nn.Parameter(ff_weights)
        self.l1.synapse_rec.weight = torch.nn.Parameter(rec_weights)
        self.l1.neuron.leak_i = torch.nn.Parameter(l1_leak_i)
        self.l1.neuron.leak_v = torch.nn.Parameter(l1_leak_v)
        # self.l1.neuron.thresh = torch.nn.Parameter(l1_threshold)
        self.p_out.synapse.weight = torch.nn.Parameter(out_weights)


    def mask_grads(self):
        self.l1.neuron.leak_i.grad *= self.neuron_mask.to(self.l1.neuron.leak_i.device)
        self.l1.neuron.leak_v.grad *= self.neuron_mask.to(self.l1.neuron.leak_v.device)
        self.l1.synapse_rec.weight.grad *= self.neuron_mask_rec.to(self.l1.synapse_rec.weight.device)


    def detach_hidden(self):
        for i in range(len(self.l1.state)):
            self.l1.state[i] = [state.detach() for state in self.l1.state[i]]
        for i in range(len(self.p_out.state)):
            self.p_out.state[i] = [state.detach() for state in self.p_out.state[i]]


class FullRSNN(BaseSNN):
    """
    Full RSNN that can be initialized from 2 separate models.
    """

    def __init__(self, l1, l2, l3, p_out):
        super().__init__()
        self.l1 = LinearCubaLif(**l1)
        self.l2 = LinearRnnCubaLif(**l2)
        self.l3 = LinearRnnCubaLif(**l3)
        self.p_out = LinearIdentity(**p_out)
        self.layers = torch.nn.ModuleList([self.l1, self.l2, self.l3, self.p_out])

    def forward(self, input):
        '''
        Forward pass through the network.
        For full network, pass through last layer twice
        '''
        for layer_id, layer in enumerate(self.layers):
            if layer_id == 2:
                layer(input)
            input = layer(input)
            # if len(layer.state) == 3:
            #     # print(f"in, i, v, s, th: {input[0, 0]},{layer.state[2][0][0, 0]}, {layer.state[2][1][0, 0]}, {layer.state[2][2][0, 0]}, {layer.neuron.thresh[0]}")
            #     print(layer.state[2][2][0, :])
            # if len(layer.state) == 2:
            #     if len(layer.state[1]) == 3:
            #         # print(f"in, i, v, s, th: {input[0, 0]}, {layer.state[1][0][0, 0]}, {layer.state[1][1][0, 0]}, {layer.state[1][2][0, 0]}, {layer.neuron.thresh[0]}")
            #         print(layer.state[1][2][0, :])
        return input

    def load_combined_state_dict(self, state_dict_rate, state_dict_torque):
        own_state = self.state_dict()
        for name, param in state_dict_rate.items():
            if name not in own_state:
                 continue
            if name == "p_out.synapse.weight":
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
        for name, param in state_dict_torque.items():
            name = name.replace("l1", "l3")
            if name == "l3.synapse_ff.weight":
                param = param @ state_dict_rate["p_out.synapse.weight"]
            if name not in own_state:
                 continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)