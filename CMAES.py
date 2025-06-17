from collections.abc import Callable
import numpy as np
import math

import cma
import torch

class SurrogateCMAES:
    """
    Implements the CMA-ES algorithm with surrogate model utilization

    Attributes
    ----------
    dim : int
        The dimensions of the input to the optimized function
    eval_budget : int
        The number of maximum function evaluations per dimension
    device : str
        The device on which to train/evaluate the surrogate model. Accepts 'cuda' or 'cpu'
    archive_X : torch.Tensor
        Tensor storing original-evaluated candidates
    archive_y : torch.Tensor
        Tensor storing the fitness of original-evaluated candidates
    cmaes : cma.CMAEvolutionStrategy
        The initialized algorithm for sampling candidates and updating the parameters of CMA-ES
    """
    def __init__(self, dim: int, initial_mean: list[float] | float = 0.0, initial_step_size: float = 0.3,
                 pop_size: int | None = None, eval_budget_per_dim: int = 250, seed: int = 123, device: str = 'cuda') -> None:
        """
        
        Parameters
        ----------
        dim : int
            The dimensions of the input to the optimized function
        initial_mean : list or float
            The starting point for the search. If a single number is provided, it is assumed to be constant accross dim
        initial_step_size : float
            The starting step size for the algorithm
        pop_size : int, optional
            The number of samples to draw during each generation. If not specified, set to 4+math.floor(3*math.log(input_dim))
        eval_budget_per_dim : int
            The number of maximum function evaluations per dimension
        seed : int
            Seed for reproducibility
        """
        # Store the parameters
        self.dim: int = dim
        self.eval_budget: int = eval_budget_per_dim * dim
        self.device: str = device

        # Initialize the archive
        self.archive_X: torch.Tensor = torch.empty(self.eval_budget, self.dim, device=self.device)
        self.archive_y: torch.Tensor = torch.empty(self.eval_budget, 1, device=self.device)

        # Validate the initial search point
        if isinstance(initial_mean, float):
            initial_mean = [initial_mean] * dim
        elif dim != len(initial_mean):
            raise Exception(f'The specified dimension ({dim}) does not correspond to the dimension of the initial search point ({len(initial_mean)})')

        # Set options for the algorithm
        options: cma.CMAOptions = cma.CMAOptions()
        options.set('seed', seed)
        if pop_size:
            options.set('popsize', pop_size)

        # Initialize the algorithm
        self.cmaes: cma.CMAEvolutionStrategy = cma.CMAEvolutionStrategy(x0=initial_mean, sigma0=initial_step_size, options=options)

    def optimize_simple(self, func: Callable[[float], float], get_surrogate: Callable[..., Callable[[float], float]], g_m: int = 5,
                        f_opt: float | None = None, tol: float = 1e-8, verbose: bool = False) -> float:
        """
        Method optimizing the given function using the initialized algorithm. The surrogate model utilization follows S-CMA-ES

        Parameters
        ----------
        func : callable
            Function to be optimized
        get_surrogate : callable
            Function initiating and training the given surrogate for each iteration. Returns the trained model which can be used for predictions
        g_m : int
            Number of model-evaluated generations
        f_opt : float, optional
            The target optimal function value to reach
        tol : float
            The tolerance for reaching the optimal function value
        verbose : bool
            Indicator whether to print progress to the console

        Returns
        -------
        float
            The optimal point found by the optimization
        """
        # Intialize variables
        evals: int = 0 # The number of performed evaluations using the original function
        gen: int = 0 # Current generation number
        f_best: float = math.inf # The best function value found so far

        while evals < self.eval_budget: # Loop until the evaluation budget is reached
            # Sample the candidates 
            candidates: list = self.cmaes.ask() 
            candidates_tensor: torch.Tensor = torch.tensor(np.array(candidates), device=self.device, dtype=torch.float32) # Move the candidates to the device where the model resides

            # Original-evaluated generations
            if (gen % g_m == 0) or (evals < 1000):
                fitnesses: list = [func(x) for x in candidates[:min(self.cmaes.popsize, self.eval_budget - evals)]] # Evaluate candidates using the original function (for the last generation, use only as much evaluations as allowed by the budget)
                
                # Store data to archive
                self.archive_X[evals:(evals+len(fitnesses)), :] = candidates_tensor
                self.archive_y[evals:(evals+len(fitnesses)), 0] = torch.tensor(fitnesses, device=self.device)
                evals += len(fitnesses) # Add the number of original-fitness evaluations
    
                # Obtain the trained surrogate
                surrogate_model: Callable[[float], float] = get_surrogate(self.dim, self.archive_X[:evals, :], self.archive_y[:evals, :], device=self.device)

                # Check best current fitness
                f_best_g: float = min(fitnesses)
                if f_best > f_best_g:
                    f_best = f_best_g

                # Stop if the desired value was reached
                if (f_opt is not None) and (f_best + tol <= f_opt):
                    break

            # Model-evaluated generations
            else:
                fitnesses: list = surrogate_model(candidates_tensor)[:, 0].tolist()
                
            # Update CMA-ES
            self.cmaes.tell(candidates, fitnesses)
            gen += 1

            # Print progress
            if verbose:
                print(f'Generation: {gen}. Evaluation: {evals}/{self.eval_budget}. Best fitness: {f_best}                     ')

        # Print information about best obtained fitness
        if verbose:
            print(f'Best found value: {f_best}. Optimal value: {f_opt}')

        # Return the best found value
        return f_best

    def optimize_dts(self, func: Callable[[float], float], get_surrogate: Callable[..., Callable[[float], float]], orig_ratio: float = 0.05,
                        f_opt: float | None = None, tol: float = 1e-8, verbose: bool = False) -> float:
        """
        Method optimizing the given function using the initialized algorithm. The surrogate model utilization follows DTS-CMA-ES

        Parameters
        ----------
        func : callable
            Function to be optimized
        get_surrogate : callable
            Function initiating and training the given surrogate for each iteration. Returns the trained model which can be used for predictions
        orig_ratio : float
            The ratio of original-evaluated points during each iteration
        f_opt : float, optional
            The target optimal function value to reach
        tol : float
            The tolerance for reaching the optimal function value
        verbose : bool
            Indicator whether to print progress to the console

        Returns
        -------
        float
            The optimal point found by the optimization
        """
        # Intialize variables
        evals: int = 0 # The number of performed evaluations using the original function
        gen: int = 0 # Current generation number
        f_best: float = math.inf # The best function value found so far

        while evals < self.eval_budget: # Loop until the evaluation budget is reached
            # Sample the candidates 
            candidates: list = self.cmaes.ask() 
            candidates_tensor: torch.Tensor = torch.tensor(np.array(candidates), device=self.device, dtype=torch.float32) # Move the candidates to the device where the model resides

            # Train the first model
            rankings_tensor: torch.Tensor # List of the candidate rankings according to their fitness (on self.device)
            rankings: list # List of the candidate rankings according to their fitness
            if evals > 0: # Train only if there are observations available (skips the first generation)
                surrogate_model1: Callable[[float], float] = get_surrogate(self.dim, self.archive_X[:evals, :], self.archive_y[:evals, :], device=self.device)
                rankings_tensor = surrogate_model1(candidates_tensor)[:, 0].argsort()
                rankings = rankings_tensor.tolist()
            else: # If no observations are in the archive, choose the observations to evaluate at random
                rankings = list(range(len(candidates)))
                rankings_tensor = torch.tensor(rankings, device=self.device)
            
            # Evaluate a ratio of candidates with orginal fitness
            orig_cutoff: int = min(math.ceil(orig_ratio*len(candidates)), self.eval_budget-evals) # Get the number of candidates to be orignally-evaluated (make sure the evaluation budget is not surpassed)
            fitnesses: list[float] = [func(candidates[idx]) for idx in rankings if idx < orig_cutoff]

            # Check best current fitness
            f_best_g: float = min(fitnesses)
            if f_best > f_best_g:
                f_best = f_best_g

            # Stop if the desired value was reached
            if (f_opt is not None) and (f_best + tol <= f_opt):
                break

            # Add the evaluated candidates to the archive
            self.archive_X[evals:(evals+len(fitnesses)), :] = candidates_tensor[rankings_tensor < orig_cutoff]
            self.archive_y[evals:(evals+len(fitnesses)), 0] = torch.tensor(fitnesses, device=self.device)
            evals += len(fitnesses)

            # Train the second model
            surrogate_model2: Callable[[float], float] = get_surrogate(self.dim, self.archive_X[:evals, :], self.archive_y[:evals, :], device=self.device)

            # Evaluate fitness of non-original-evaluated points
            fitnesses_surrogate: list = surrogate_model2(candidates_tensor[rankings_tensor >= orig_cutoff])[:, 0].tolist()

            # Prepare candidates and fitnesses
            final_fitnesses: list = fitnesses + fitnesses_surrogate # Combine original-evaluated and estimated fitnesses
            final_candidates: list = [candidates[idx] for idx in rankings] # Reorder candidates to match fitnesses
                
            # Update CMA-ES
            self.cmaes.tell(final_candidates, final_fitnesses)
            gen += 1

            # Print progress
            if verbose:
                print(f'Generation: {gen}. Evaluation: {evals}/{self.eval_budget}. Best fitness: {f_best}                     ')

        # Print information about best obtained fitness
        if verbose:
            print(f'Best found value: {f_best}. Optimal value: {f_opt}')

        # Return the best found value
        return f_best