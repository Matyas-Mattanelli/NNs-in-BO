import cocoex
from NTKSurrogate import NTKSurrogate, create_NTKSurrogate_generator
from CMAES import SurrogateCMAES
from collections.abc import Callable

if __name__ == '__main__':
    # Input CMA-ES
    input_dim: int = 10 # Dimension of the input
    initial_mean: int = 0.0
    initial_step_size: float = 0.3
    pop_size: int | None = None # If None, the standard size recommended by Hansen (2006) is used - 4+math.floor(3*math.log(input_dim))
    evaluation_budget: int = 250 # Number of allowed function evaluations per dimension
    g_m: int = 5 # Number of model-evaluated generations
    seed: int = 123
    verbose_cmaes: bool = True
    
    # Input (surrogate)
    hidden_dim: int = 5_000 # Number of nodes in the hidden layer (follows Pariah et al. (2022))
    device: str = 'cuda' # Whether to train the model on GPU or CPU
    batch_size: int = None # Batch size for model training (None for all observations to be used during each gradient update)
    n_epochs: int = 100 # Number of training iterations for the surrogate
    lr: float = 0.001 # Learning rate
    alpha: float = 1.0 # Regularization parameter
    gamma: float = 1.0 # Variance of the normal distribution from which weights are initialized
    verbose_surrogate: bool = False
    validate: bool = True
    
    # Get the simplest BBOB function
    problem: cocoex.BareProblem = cocoex.BareProblem(suite_name="bbob", function=1, dimension=input_dim, instance=1)

    # Initialize the optimizer and surrogate
    optimizer: SurrogateCMAES = SurrogateCMAES(dim=input_dim, initial_mean=initial_mean, initial_step_size=initial_step_size, pop_size=pop_size,
                                               eval_budget_per_dim=evaluation_budget, seed=seed, device=device)
    get_NTKSurrogate: Callable[..., NTKSurrogate] = create_NTKSurrogate_generator(input_params={'hidden_dim':hidden_dim,'gamma':gamma},
                                                                                  train_params={'n_epochs':n_epochs, 'batch_size':batch_size,
                                                                                                'alpha':alpha, 'verbose':verbose_surrogate,
                                                                                                'lr':lr, 'validate':validate})
    
    # Optimize S-CMA-ES
    res: float = optimizer.optimize_simple(func=problem, get_surrogate=get_NTKSurrogate, g_m=g_m, f_opt=problem.best_value(), verbose=verbose_cmaes)

    # Optimize DTS-CMA-ES
    #res: float = optimizer.optimize_dts(func=problem, get_surrogate=get_NTKSurrogate, orig_ratio=0.05, f_opt=problem.best_value(), verbose=verbose_cmaes)