# Neural network with Neural Tangent Kernel (NTK) parametrization
# Follows Paria et al. 2022
# 1 hidden layer, 5000 nodes
# NTK: Weights initialized from a normal ditribution with zero mean and gamma/5000 variance
# NTK: Biases initialized from a normal distribution with zero mean and gamma variance (except for output layer where the bias is initialized to 0)
# NTK: Activation function tanh
# Learning rate 0.001, Adam optimizer (should be SGD but Adam makes no difference apart from faster convergence)
# MSE loss + regularization parameter (distance from initialization - Paria 2022) with strenth alpha

import torch
import math
from collections.abc import Callable
from utils import CustomDataset

class NTKSurrogate(torch.nn.Module):
    """
    Neural network with Neural Tangent Kernel parametrization

    Attributes
    ----------
    input_dim : int
        The dimensions of the input
    hidden_dim : int
        Number of nodes in the hidden layer
    gamma : float
        The variance of the normal distribution from which weights are initialized
    hidden_layer : torch.nn.Linear
        Object containing the hidden layer
    output_layer : torch.nn.Linear
        Object containing the output layer
    device : str
        The device on which the model will be trained on (either 'cuda' or 'cpu')
    """
    def __init__(self, input_dim: int, hidden_dim: int, gamma: float = 1.0, device: str = 'cuda') -> None:
        """
        Runs the constructor of torch.nn.Module and initializes the layers for the neural network

        Parameters
        ----------
        input_dim : int
            The dimensions of the input
        hidden_dim : int
            The number of nodes in the hidden layer
        gamma : float
            The variance of the normal distribution from which weights are initialized
        device : str
            The device on which the model will be trained on (either 'cuda' or 'cpu')
        """
        # Run the constructor for the base torch module
        super().__init__() 

        # Define the layers
        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.hidden_layer: torch.nn.Linear = torch.nn.Linear(input_dim, hidden_dim)
        self.output_layer: torch.nn.Linear = torch.nn.Linear(hidden_dim, 1)
        
        # Perform NTK initialization
        self.gamma: float = gamma
        self._NTK_init()

        # Move to the correct device
        self.to(device)
        self.device: str = device

        # Store initial parameters
        self.initial_params = torch.cat([p.detach().flatten() for p in self.parameters()])

        # Initialize mean and standard deviation for standardization
        self.mean: float = torch.zeros(input_dim, device=device)
        self.st_dev: float = torch.ones(input_dim, device=device)


    def _NTK_init(self) -> None:
        """
        NTK intializer for the weights and biases in the network.
        """
        # Initialize the weights from the normal distribution with zero mean and gamma/layer_input_size variance (Paria 2022)
        torch.nn.init.normal_(self.hidden_layer.weight, mean=0.0, std=self.gamma/self.input_dim)
        torch.nn.init.normal_(self.output_layer.weight, mean=0.0, std=self.gamma/self.hidden_dim)

        # Initialize the biases from the normal distribution with zero mean and 1 variance
        torch.nn.init.normal_(self.hidden_layer.bias, mean=0.0, std=1.0)
        
        # Initialize the bias in the output layer to zero
        torch.nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Method to be run during the forward pass of the network

        Parameters
        ----------
        X : torch.Tensor
            Tensor containing the data to be propagated through the network

        Returns
        -------
        torch.Tensor
            The Tensor resulting from the propagation of the data through the netwok
        """
        # Hidden layer
        X = torch.tanh(self.hidden_layer(X))
        
        return self.output_layer(X)
    
    def train_loop(self, X: torch.Tensor, y: torch.Tensor, n_epochs: int = 100, batch_size: int | None = 64, loss_name: str = 'MSE',
                   alpha: float = 1.0, optimizer_name: str = 'Adam', lr: float = 0.001, validate: bool = False,
                   val_ratio: float = 0.1, verbose: bool = False) -> None:
        """
        Trains the model on the given data set for a given number of epochs

        Parameters
        ----------
        X : torch.Tensor
            Training data (input variables)
        y : torch.Tensor
            Training data (output variable)
        n_epochs : int
            Number of iterations of the training loop
        batch_size : int or
            Number of observations to consider during each gradient update. If None, all observations are used during a single update
        loss_name : str
            Name of the loss function to use for optimization. Currently supports only 'MSE'
        alpha : float
            Number controlling the regularization strength
        optimizer_name : str
            Name of the optimizer to use. Currently supports: ['Adam', 'SGD']
        lr : float
            Learning rate for the optimizer
        validate : bool
            Indicator whether to split the data into train and validation parts and verify model performance on validation
        val_ratio : float
            The portion of data to use for validation purposes
        verbose : bool
            Indicator whether to display progress in the console
        """
        # Define the loss function
        match loss_name:
            case 'MSE':
                loss_fn: torch.nn.MSELoss = torch.nn.MSELoss(reduction='mean')
            case _:
                raise Exception(f'The loss {loss_name} is not supported')
            
        # Define the optimizer
        match optimizer_name:
            case 'Adam':
                optimizer: torch.optim.Adam = torch.optim.Adam(self.parameters(), lr=lr)
            case 'SGD':
                optimizer: torch.optim.SGD = torch.optim.SGD(self.parameters(), lr=lr)
            case _:
                raise Exception(f'The optimizer {optimizer_name} is not supported')
        
        # Separate train and validation sets if required
        if validate:
            sep_idx: int = math.ceil(X.shape[0]*(1-val_ratio))
            X_train: torch.Tensor = X[:sep_idx, :]
            y_train: torch.Tensor = y[:sep_idx, :]
            X_val: torch.Tensor = X[sep_idx:, :]
            y_val: torch.Tensor = y[sep_idx:, :]
        else:
            X_train, y_train = X, y

        # Create the datasets
        batch_size = batch_size if batch_size is not None else X_train.shape[0]
        dataloader_train: torch.utils.data.DataLoader = torch.utils.data.DataLoader(CustomDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
        if validate:
            dataloader_val: torch.utils.data.DataLoader = torch.utils.data.DataLoader(CustomDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        # Set the model to training mode
        self.train()

        # Loop through the epochs
        for epoch in range(n_epochs):

            # Training
            for X, y in dataloader_train:
                # Compute prediction and loss
                preds: torch.Tensor = self(X)
                loss: torch.Tensor = loss_fn(preds, y)

                # Add regularization if required
                if alpha > 0.0:
                    current_params = torch.cat([p.flatten() for p in self.parameters()]) # Get the current weights
                    loss += alpha*torch.sum((current_params - self.initial_params) ** 2)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Validation
            val_str: str
            if validate:
                self.eval()
                with torch.no_grad():
                    loss_val: torch.Tensor = torch.tensor(0.0, device=self.device)
                    for X, y in dataloader_val:
                        preds_val: torch.Tensor = self(X)
                        loss_val += loss_fn(preds_val, y)
                self.train()
                val_str = f' Validation loss: {loss_val.item():.5f}'
            else:
                val_str = ''

            # Display progress
            if verbose:
                print(f'Epoch: {epoch} Loss: {loss.item():.5f}' + val_str)

        # Set to evaluation mode after finishing training
        self.eval()

def create_NTKSurrogate_generator(input_params: dict, train_params: dict) -> Callable[..., NTKSurrogate]:
    """
    Creates a function which initiates and trains an NTKSurrogate for a given set of parameters

    Parameters
    ----------
    input_params : dict
        Dictionary of parameters to be passed to NTKSurrogate during initialization
    train_params : dict
        Dictionary of parameters to be passed to NTKSurrogate during training

    Returns
    -------
    callable
        Function returning a trained instance of NTKSurrogate
    """
    def generate_NTKSurrogate(input_dim: int, X: torch.Tensor, y: torch.Tensor, device: str) -> NTKSurrogate:
        """
        Initiates and trains an instance of NTKSurrogate

        Parameters
        ----------
        input_dim : int
            The dimensions of the input
        X : torch.Tensor
            Training data (input variables)
        y : torch.Tensor
            Training data (output variable)

        Returns
        -------
        NTKSurrogate
            Trained instance of NTKSurrogate
        """
        # Initiate the model
        model: NTKSurrogate = NTKSurrogate(input_dim=input_dim, device=device, **input_params)

        # Train the model
        model.train_loop(X, y, **train_params)

        return model

    return generate_NTKSurrogate

