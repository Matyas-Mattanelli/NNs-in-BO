import pfns4bo
import torch
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod

def get_PFNSurrogate(device: str, model_path: str = pfns4bo.hebo_plus_model):
    """
    Function loading a pretrained PFN model

    Parameters
    ----------
    device : str
        The device on which inference will be performed (either 'cuda' or 'cpu')
    model_path : str
        Path to the pretrained model to be loaded via torch

    Returns
    -------
    pfns4bo.TransformerBOMethod
        Instance of the pretrained model
    """
    pfn_bo: TransformerBOMethod = TransformerBOMethod(torch.load(model_path), device=device)

    return pfn_bo