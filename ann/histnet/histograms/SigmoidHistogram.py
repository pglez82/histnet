import torch

class SigmoidHistogram(torch.nn.Module):
    """
    This class allows to build a histogram that is differentiable. This histogram is an aproximation, not the real one
    Following: https://discuss.pytorch.org/t/differentiable-torch-histc/25865
    """
    def __init__(self, num_bins, min, max, sigma,n_examples_bag,quantiles=False):
        super(SigmoidHistogram, self).__init__()
        self.num_bins = num_bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.n_examples_bag = n_examples_bag
        self.quantiles = quantiles
        self.delta = float(max - min) / float(num_bins) #Compute the width of each bin
        self.register_buffer('centers',float(min) + self.delta * (torch.arange(num_bins).float() + 0.5)) #Compute the center of each bin
        

    def forward(self, input):
        """Function that computes the histogram. It is prepared to compute multiple histograms at the same time, saving a lot of time.
            Args:
                x Tensor. Two dimension tensor. Each row should be all possible values for a single feature (as many values as columns). 
            Returns
                A vector with size n_features*n_bins, containing all the histograms
        """
        input = torch.unsqueeze(input.transpose(0,1), 1) - torch.unsqueeze(self.centers, 1) #compute the distance to the center
        input = torch.sigmoid(self.sigma * (input + self.delta/2)) - torch.sigmoid(self.sigma * (input - self.delta/2))
        input = input.flatten(end_dim=1).sum(dim=1)/self.n_examples_bag

        if self.quantiles:
            input = input.view(-1,self.num_bins).cumsum(dim=1)
        
        return input.flatten()