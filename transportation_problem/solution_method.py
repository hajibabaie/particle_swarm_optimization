import copy
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
            self.cost = np.inf
            self.solution = None

    def __init__(self,
                 cost_function,
                 max_iteration,
                 variables_shape,
                 min_range_of_variables,
                 max_range_of_variables,
                 number_of_particles,
                 inertia_rate,
                 inertia_damping_rate,
                 personal_learning_rate,
                 global_learning_rate):

        self._cost_function = cost_function
        self._max_iteration = max_iteration
        self._variables_shape = variables_shape
        self._min_range_of_variables = min_range_of_variables
        self._max_range_of_variables = max_range_of_variables
        self._max_velocity = 0.1 * (self._max_range_of_variables - self._min_range_of_variables)
        self._min_velocity = -1 * self._max_velocity
        self._number_particles = number_of_particles
        self._particles = [self._Particle() for _ in range(self._number_particles)]
        self._particle_best = self._ParticleBest()
        self._inertia_rate = inertia_rate
        self._inertia_damping_rate = inertia_damping_rate
        self._personal_learning_rate = personal_learning_rate
        self._global_learning_rate = global_learning_rate
        self._best_costs = []

    def _initialization(self):

        for i in range(self._number_particles):

            self._particles[i].position = np.random.uniform(self._min_range_of_variables,
                                                            self._max_range_of_variables,
                                                            self._variables_shape)

            self._particles[i].velocity = np.zeros_like(self._particles[i].position)

            self._particles[i].solution, self._particles[i].cost = \
            self._cost_function(self._particles[i].position)

            self._particles[i].position_best = np.copy(self._particles[i].position)
            self._particles[i].cost_best = self._particles[i].cost.copy()

            if self._particles[i].cost_best < self._particle_best.cost:

                self._particle_best.position = np.copy(self._particles[i].position_best)
                self._particle_best.cost = self._particles[i].cost_best.copy()
                self._particle_best.solution = copy.deepcopy(self._particles[i].solution)

    def _update_velocity(self):

        for i in range(self._number_particles):

            self._particles[i].velocity = self._inertia_rate * self._particles[i].velocity + \
            self._personal_learning_rate * np.random.random(self._particles[i].velocity.shape) * \
            (self._particles[i].position_best - self._particles[i].position) + \
            self._global_learning_rate * np.random.random(self._particles[i].velocity.shape) * \
            (self._particle_best.position - self._particles[i].position)

            self._particles[i].velocity = np.clip(self._particles[i].velocity, self._min_velocity, self._max_velocity)

            self._particles[i].position = self._particles[i].position + self._particles[i].velocity

            self._particles[i].position = np.clip(self._particles[i].position,
                                                  self._min_range_of_variables, self._max_range_of_variables)

            self._particles[i].solution, self._particles[i].cost = self._cost_function(self._particles[i].position)

            if self._particles[i].cost < self._particles[i].cost_best:

                self._particles[i].position_best = np.copy(self._particles[i].position)
                self._particles[i].cost_best = self._particles[i].cost.copy()

                if self._particles[i].cost_best < self._particle_best.cost:

                    self._particle_best.position = np.copy(self._particles[i].position_best)
                    self._particle_best.cost = self._particles[i].cost.copy()
                    self._particle_best.solution = copy.deepcopy(self._particles[i].solution)


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
        plt.title("Transportation Problem Using Particle Swarm Optimization", fontweight="bold")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.savefig("./cost_function.png")

        return self._particles, self._particle_best, toc - tic
