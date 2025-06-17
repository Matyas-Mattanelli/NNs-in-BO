import torch

class CustomDataset(torch.utils.data.Dataset):
    """
    Custom wrapper for the torch Dataset class
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Stores the input data

        Parameters
        ----------
        X : torch.Tensor
            Training data (input variables)
        y : torch.Tensor
            Testing data (output variable)
        """
        self.X: torch.Tensor = X
        self.y: torch.Tensor = y

    def __len__(self) -> int:
        """
        Returns the size of the dataset

        Returns
        -------
        int
            Size of the dataset
        """
        return self.X.shape[0]
    
    def __getitem__(self, idx:  int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the observation at the given index

        Parameters
        ----------
        idx : int
            Index of the required observation

        Returns
        -------
        tuple
            The input and output observation at the given index
        """
        return self.X[idx, :], self.y[idx, :]