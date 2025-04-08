
import torch
import numpy as np
import matplotlib.pyplot as plt

from evotorch import Problem, Solution
from evotorch.algorithms import GeneticAlgorithm, CMAES
from evotorch.logging import StdOutLogger
from evotorch.operators import GaussianMutation, SimulatedBinaryCrossOver

class CrazyflieProblem(Problem):
    def __init__(self, trainer):
        super().__init__(
            objective_sense="min",  # the goal is to minimize
            solution_length=trainer.solution_size,
            initial_bounds=(-0.2, 0.2),
            dtype=torch.float32,
            num_actors=1
        )
        self.trainer = trainer
        self.seed = 235903750

    def set_random_seed_across_all_problems(self):
        np.random.seed()
        seed = np.random.randint(10000000)
        self.all_remote_problems().set_seed(seed)
        # print(f"new seed: {seed}")

    def set_seed(self, seed):
        # print(f"set new seed: {seed}")
        self.seed = seed
    
    def _evaluate(self, solution: Solution):
        # print(f"evaluate: {self.seed}")
        np.random.seed(self.seed)
        # Compute the fitness
        fitness = self.trainer.obj_func(solution.values)

        # Register the fitness into the Solution object
        solution.set_evaluation(fitness)


class Trainer:
    def __init__(self, model, train_loader, loss_function, config):
        self.model = model
        self.train_loader = train_loader
        self.loss_function = loss_function
        self.device = config["device"]

        # Determine indices of model parameters in solution vector
        self.i_ff_weights = model.l1.synapse_ff.weight.numel()
        self.i_rec_weights = self.i_ff_weights + model.l1.synapse_rec.weight.numel()
        self.i_l1_leak_i = self.i_rec_weights + model.l1.neuron.leak_i.numel()
        self.i_l1_leak_v = self.i_l1_leak_i + model.l1.neuron.leak_v.numel()
        self.i_l1_threshold = self.i_l1_leak_v + model.l1.neuron.thresh.numel()
        self.i_out_weights = self.i_l1_threshold + model.p_out.synapse.weight.numel()

        self.solution_size = self.i_ff_weights + self.i_rec_weights + self.i_l1_leak_i + self.i_l1_leak_v + self.i_l1_threshold + self.i_out_weights

    def set_model_weights(self, solution):
        ff_weights = solution[:self.i_ff_weights].reshape(self.model.l1.synapse_ff.weight.shape).clone()
        rec_weights = solution[self.i_ff_weights:self.i_rec_weights].reshape(self.model.l1.synapse_rec.weight.shape).clone()
        l1_leak_i = solution[self.i_rec_weights:self.i_l1_leak_i].reshape(self.model.l1.neuron.leak_i.shape).clone()
        l1_leak_v = solution[self.i_l1_leak_i:self.i_l1_leak_v].reshape(self.model.l1.neuron.leak_v.shape).clone()
        l1_threshold = solution[self.i_l1_leak_v:self.i_l1_threshold].reshape(self.model.l1.neuron.thresh.shape).clone()
        out_weights = solution[self.i_l1_threshold:self.i_out_weights].reshape(self.model.p_out.synapse.weight.shape).clone()

        self.model.set_weights_from_vectors(ff_weights, rec_weights, l1_leak_i, l1_leak_v, l1_threshold, out_weights)
        

    def evaluate(self, solution):
        self.set_model_weights(solution)
        self.model.reset()
        self.model.eval()

        # data, target = next(iter(self.train_loader))
        # data, target = data.to(self.device), target.to(self.device)
        # data = data.permute(1, 0, 2)
        # target = target.permute(1, 0, 2)

        output = self.model.forward_sequence(self.data)
        return output, self.target

    # Objective function with mse to target as error
    def obj_func(self, solution):
        # global random_seed
        # np.random.seed(random_seed)
        output, target = self.evaluate(solution)
        fitness = torch.nn.functional.mse_loss(output, target)
        # print(fitness)
        return fitness

    def set_data(self):
        data, target = next(iter(self.train_loader))
        data, target = data.to(self.device), target.to(self.device)
        self.data = data.permute(1, 0, 2)
        self.target = target.permute(1, 0, 2)
        
def fit(model, train_loader, val_loader, optimizer, config, out_dir, loss_function):
    trainer = Trainer(model, train_loader, loss_function, config)
    problem = CrazyflieProblem(trainer)

    # Run a GA similar to NSGA-II for 100 steps and log results to standard output every 1 steps
    searcher = GeneticAlgorithm(
        problem,
        popsize=40,
        operators=[
            SimulatedBinaryCrossOver(problem, tournament_size=4, cross_over_rate=1.0, eta=8),
            GaussianMutation(problem, stdev=0.1),
        ],
    )


    # searcher = CMAES(
    #     problem, 
    #     stdev_init=0.04, # reset models
    #     popsize=10
    # )

    searcher.before_step_hook.append(trainer.set_data)
    
    logger = StdOutLogger(searcher, interval=1)

    def plot_best():
        best = searcher.status['pop_best']
        trainer.set_model_weights(best.values)
        trainer.model.reset()
        trainer.model.eval()

        output = trainer.model.forward_sequence(trainer.data)
        plt.plot(output[:, 0].detach().cpu().numpy())
        plt.plot(trainer.target[:, 0].detach().cpu().numpy())
        plt.show()

    searcher.after_step_hook.append(plot_best)
    searcher.run(200)

    print("Final status:\n", searcher.status)
    est, target = trainer.evaluate(searcher.status['pop_best'].values)

    fitness = torch.nn.functional.mse_loss(est, target)
    print(fitness)

    
    # return best_model, best_fitness, epoch, avg_time, mean_losses, test_losses








