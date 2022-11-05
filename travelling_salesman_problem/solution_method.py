import numpy as np
import matplotlib.pyplot as plt
import time


class PSO:

    class _Particle:

        def __init__(self):

            self.position = None
            self.position_best = None
            self.velocity = None
            self.cost = None
            self.cost_best = None
            self.solution = None

    class _ParticleBest:

        def __init__(self):

            self.position = None
            self.solution = None
            self.cost = np.inf

    def __init__(self,
                 cost_function,
                 max_iteration,
                 number_of_particles,
                 number_of_variables,
                 min_range_of_variables,
                 max_range_of_variables,
                 inertia_rate,
                 inertia_damping_rate,
                 personal_learning_rate,
                 global_learning_rate):

        self._cost_function = cost_function
        self._max_iteration = max_iteration
        self._number_of_particles = number_of_particles
        self._number_of_variables = number_of_variables
        self._min_range_of_variables = min_range_of_variables
        self._max_range_of_variables = max_range_of_variables
        self._max_velocity = 0.1 * (self._max_range_of_variables - self._min_range_of_variables)
        self._min_velocity = -1 * self._max_velocity
        self._inertia_rate = inertia_rate
        self._inertia_damping_rate = inertia_damping_rate
        self._personal_learning_rate = personal_learning_rate
        self._global_learning_rate = global_learning_rate
        self._particles = [self._Particle() for _ in range(self._number_of_particles)]
        self._particle_best = self._ParticleBest()
        self._best_cost = []

    def _initialization(self):

        for i in range(self._number_of_particles):

            self._particles[i].position = np.random.uniform(self._min_range_of_variables,
                                                            self._max_range_of_variables,
                                                            (1, self._number_of_variables))

            self._particles[i].velocity = np.zeros_like(self._particles[i].position)

            self._particles[i].solution, \
            self._particles[i].cost = self._cost_function(self._particles[i].position)

            self._particles[i].position_best = np.copy(self._particles[i].position)
            self._particles[i].cost_best = self._particles[i].cost.copy()

            if self._particles[i].cost_best < self._particle_best.cost:

                self._particle_best.position = np.copy(self._particles[i].position)
                self._particle_best.cost = self._particles[i].cost.copy()
                self._particle_best.solution = np.copy(self._particles[i].solution)

    def _update_velocity(self):

        for i in range(self._number_of_particles):

            self._particles[i].velocity = self._inertia_rate * self._particles[i].velocity + \
            (np.random.uniform(self._min_range_of_variables,
                               self._max_range_of_variables,
                               (1, self._number_of_variables)) * self._personal_learning_rate) * \
            (self._particles[i].position_best - self._particles[i].position) + \
            (np.random.uniform(self._min_range_of_variables,
                               self._max_range_of_variables,
                               (1, self._number_of_variables)) * self._global_learning_rate) * \
            (self._particle_best.position - self._particles[i].position)

            self._particles[i].velocity = np.clip(self._particles[i].velocity, self._min_velocity, self._max_velocity)

            self._particles[i].position = self._particles[i].position + self._particles[i].velocity

            self._particles[i].position = np.clip(self._particles[i].position, self._min_range_of_variables,
                                                  self._max_range_of_variables)

            self._particles[i].solution, \
            self._particles[i].cost = self._cost_function(self._particles[i].position)

            new_solution_particles = self._new_tour(self._particles[i].position)

            solution, cost = self._cost_function(new_solution_particles)

            if cost < self._particles[i].cost:

                self._particles[i].position = np.copy(new_solution_particles)
                self._particles[i].cost = cost.copy()
                self._particles[i].solution = np.copy(solution)

            if self._particles[i].cost < self._particles[i].cost_best:

                self._particles[i].position_best = np.copy(self._particles[i].position)
                self._particles[i].cost_best = self._particles[i].cost.copy()

                if self._particles[i].cost_best < self._particle_best.cost:

                    self._particle_best.position = np.copy(self._particles[i].position_best)
                    self._particle_best.cost = self._particles[i].cost.copy()
                    self._particle_best.solution = np.copy(self._particles[i].solution)

                    new_solution_global = self._new_tour(self._particle_best.position)
                    solution_global, solution_cost = self._cost_function(new_solution_global)
                    if solution_cost < self._particle_best.cost:

                        self._particle_best.position = np.copy(new_solution_global)
                        self._particle_best.solution = np.copy(solution_global)
                        self._particle_best.cost = solution_cost.copy()


    @staticmethod
    def _roulette_wheel_selection(probs):

        random_number = np.random.random()

        probs_cumsum = np.cumsum(probs)

        return int(np.argwhere(random_number <= probs_cumsum)[0][0])


    @staticmethod
    def _swap(x):

        tour = np.argsort(x, axis=1)

        indices = np.random.choice(range(int(tour.shape[1])), 2, replace=False)

        min_index, max_index = int(min(indices)), int(max(indices))

        new_tour = np.copy(tour)

        new_tour[0, [min_index, max_index]] = tour[0, [max_index, min_index]]

        out = np.zeros_like(x)

        out[0, new_tour] = x[0, tour]

        return out


    def _insertion(self, x):

        method = self._roulette_wheel_selection([0.5, 0.5])

        tour = np.argsort(x)

        indices = np.random.choice(range(int(tour.shape[1])), 2, replace=False)

        min_index, max_index = int(min(indices)), int(max(indices))


        if method == 0:

            new_tour = np.concatenate((tour[:, :min_index],
                                       tour[:, min_index + 1: max_index + 1],
                                       tour[:, min_index: min_index + 1],
                                       tour[:, max_index + 1:]), axis=1)
        else:

            new_tour = np.concatenate((tour[:, :min_index + 1],
                                       tour[:, max_index: max_index + 1],
                                       tour[:, min_index + 1: max_index],
                                       tour[:, max_index + 1:]), axis=1)

        out = np.zeros_like(x)
        out[0, new_tour] = x[0, tour]

        return out

    @staticmethod
    def _reversion(x):

        tour = np.argsort(x, axis=1)

        indices = np.random.choice(range(int(tour.shape[1])), 2, replace=False)

        min_index, max_index = int(min(indices)), int(max(indices))

        new_tour = np.concatenate((tour[:, :min_index],
                                   np.flip(tour[:, min_index: max_index + 1]),
                                   tour[:, max_index + 1:]), axis=1)

        out = np.zeros_like(x)

        out[0, new_tour] = x[0, tour]

        return out


    def _new_tour(self, x):

        method = self._roulette_wheel_selection([0.1, 0.2, 0.7])

        if method == 0:

            out = self._swap(x)

        elif method == 1:

            out = self._insertion(x)

        elif method == 2:

            out = self._reversion(x)

        return out

    def run(self):

        tic = time.time()

        self._initialization()

        for iter_main in range(self._max_iteration):

            self._update_velocity()

            self._inertia_rate *= self._inertia_damping_rate

            self._best_cost.append(self._particle_best.cost)

        toc = time.time()

        plt.figure(dpi=600, figsize=(10, 6))
        plt.plot(range(self._max_iteration), self._best_cost)
        plt.title("Travelling Salesman Problem Using Particle Swarm Optimization", fontweight="bold")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.savefig("./cost_function.png")

        return self._particle_best, toc - tic
