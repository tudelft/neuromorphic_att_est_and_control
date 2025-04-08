import torch

class ANN(torch.nn.Module):
    """
    TODO:
    - Record all neuron states over sequence for testing/debugging/visualization
    - Make learnable parameters a configuration
    """

    def __init__(self, input_features: int, hidden_features: int, output_features: int):
        super(ANN, self).__init__()
        
        self.do_mask_grads = None
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features

        # self.linear_relu_stack = torch.nn.Sequential(
        #     torch.nn.Linear(input_features, hidden_features, bias=False),
        #     torch.nn.Sigmoid(),
        #     torch.nn.Linear(hidden_features, hidden_features, bias=False),
        #     torch.nn.Sigmoid(),
        #     torch.nn.Linear(hidden_features, output_features, bias=False),
        #     torch.nn.Sigmoid()
        # )

        self.gru = torch.nn.RNN(input_features, hidden_features, num_layers=1, nonlinearity='relu', bias=False, batch_first=False)
        # n_ignore = 10
        # self.set_weights(n_fixed=n_ignore)
        self.output = torch.nn.Linear(hidden_features, output_features, bias=False)
        
        

    def reset(self, batch_size=1, device='cpu'):
        """
        Reset the states

        :param batch_size: desired batch_size
        :returns: A tensor containing the states all set to zero.

        TODO: Should batch size be consistent over class instantiation?
        TODO: Currently returning zeros, can initial states also be non-zero?
        """
        fake_input = torch.zeros([1, batch_size, self.input_features]).to(device)
        single_step = self.forward(fake_input)
        return single_step
    
    def set_weights(self, n_fixed, device):
        new_weights = self.gru.weight_hh_l0.clone()
        # new_weights = torch.ones_like(new_weights)
        new_weights[:n_fixed, :] = 0
        new_weights[:, :n_fixed] = 0
        # new_weights[:n_fixed, :n_fixed] = -1
        # new_weights[:n_fixed, :n_fixed] = torch.eye(n_fixed)
        new_weights[:n_fixed, :n_fixed].fill_diagonal_(1)
        self.gru.weight_hh_l0 = torch.nn.Parameter(new_weights)
        
        # self.gru.weight_hh_l0 = self.gru.weight_hh_l0 * new_weights
        # self.gru.weight_ih_l0 = torch.nn.Parameter(torch.tensor([[0.2], [-0.2], [0.2], [-0.2]]), requires_grad=False)
        
        self.rec_mask = torch.ones_like(self.gru.weight_hh_l0)
        self.rec_mask[:n_fixed, :] = 0
        self.rec_mask[:, :n_fixed] = 0

        self.to(device)

        self.do_mask_grads = True

    def forward(self, x, record=False):
        seq_length, batch_size, ninputs = x.size()
        # CHECK BATCH SIZE AGAINST INPUT?


        voltages = []

        if record:
            self.recording = []

        # self.gru.weight_ih_l0 = torch.nn.Parameter(torch.tensor([[1], [-1]], dtype=torch.float32) * 0.002)
        # self.gru.weight_hh_l0 = torch.nn.Parameter(torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32)*1)
        # new_weights = self.gru.weight_hh_l0.clone()
        # n_fixed = 4
        # # new_weights = torch.ones_like(new_weights)
        # new_weights[:n_fixed, :] = 0
        # new_weights[:, :n_fixed] = 0
        # new_weights[:n_fixed, :n_fixed] = -1
        # # new_weights[:n_fixed, :n_fixed] = torch.eye(n_fixed)
        # new_weights[:n_fixed, :n_fixed].fill_diagonal_(1)
        # self.gru.weight_hh_l0.data = new_weights
        # # self.gru.weight_hh_l0 = self.gru.weight_hh_l0 * new_weights
        # self.gru.weight_ih_l0.data = torch.tensor([[0.2], [-0.2], [0.2], [-0.2]])
        # self.output.weight.data = torch.tensor([[0.01, -0.01, 0.0, -0.0]])
        # print(self.gru.weight_ih_l0)
        # print(self.gru.weight_hh_l0)
        # print(self.output.weight)
        z, h = self.gru(x)

        # self.gru.output.weight.data = self.gru.output.weight.data * 0.02
        output = self.output(z)
        # output = torch.add(z[:, :, 0], -z[:, :, 1]).unsqueeze(2)
        # output = torch.clamp(output, min=-1, max=1)
        # output = z
#

        return output
    
    def forward_sequence(self, x):
        """
        Forward pass over a sequence of inputs

        :param x: input sequence
        :returns: output sequence
        """
        return self.forward(x)
    
    def mask_grads(self):
        """
        Backward pass

        :param grad_output: gradient of the loss with respect to the output
        :returns: gradient of the loss with respect to the input
        """
        self.gru.weight_hh_l0.grad *= self.rec_mask.to(self.gru.weight_hh_l0.grad.device)