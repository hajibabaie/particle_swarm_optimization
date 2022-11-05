import numpy as np
import matplotlib.pyplot as plt
import time


class PSO:

    class _Particle:

        def __init__(self):

            self.position = None
            self.position_best = None
            self.cost = None
            self.cost_best = None
            self.solution = None
            self.velocity = None

    class _ParticleBest:

        def __init__(self):

            self.position = None
            self.cost = np.inf
            self.solution = None

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
        self._velocity_max = 0.1 * (self._max_range_of_variables - self._min_range_of_variables)
        self._velocity_min = -1 * self._velocity_max
        self._inertia_rate = inertia_rate
        self._inertia_damping_rate = inertia_damping_rate
        self._personal_learning_rate = personal_learning_rate
        self._global_learning_rate = global_learning_rate
        self._best_costs = []
        self._particles = [self._Particle() for _ in range(self._number_of_particles)]
        self._particle_best = self._ParticleBest()

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
                self._particle_best.solution = np.copy(self._particles[i].solution)
                self._particle_best.cost = self._particles[i].cost_best.copy()

    def _update_velocity(self):

        for i in range(self._number_of_particles):

            self._particles[i].velocity = self._inertia_rate * self._particles[i].velocity + \
            (np.random.uniform(self._min_range_of_variables,
                               self._max_range_of_variables,
                               self._particles[i].velocity.shape) * self._personal_learning_rate) * \
            (self._particles[i].position_best - self._particles[i].position) + \
            (np.random.uniform(self._min_range_of_variables,
                               self._max_range_of_variables,
                               self._particles[i].velocity.shape) * self._global_learning_rate) * \
            (self._particle_best.position - self._particles[i].position)

            self._particles[i].velocity = np.clip(self._particles[i].velocity, self._velocity_min, self._velocity_max)

            self._particles[i].position = self._particles[i].position + self._particles[i].velocity

            self._particles[i].position = np.clip(self._particles[i].position, self._min_range_of_variables,
                                                  self._max_range_of_variables)

            self._particles[i].solution, \
            self._particles[i].cost = self._cost_function(self._particles[i].position)

            if self._particles[i].cost < self._particles[i].cost_best:

                self._particles[i].position_best = np.copy(self._particles[i].position)
                self._particles[i].cost_best = self._particles[i].cost.copy()

                if self._particles[i].cost < self._particle_best.cost:

                    self._particle_best.position = np.copy(self._particles[i].position)
                    self._particle_best.solution = np.copy(self._particles[i].solution)
                    self._particle_best.cost = self._particles[i].cost.copy()

    def run(self):

        tic = time.time()

        self._initialization()

        for iter_main in range(self._max_iteration):

            self._update_velocity()

            self._inertia_rate *= self._inertia_damping_rate

            self._best_costs.append(self._particle_best.cost)

        toc = time.time()

        plt.figure(dpi=600, figsize=(10, 6))
        plt.plot(range(self._max_iteration), self._best_costs)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Integer Programming Using Particle Swarm Optimization", fontweight="bold")
        plt.savefig("./cost_function")

        return self._particle_best, toc - tic
