##################################### import libraries #########################################
from models.network import EVOLUNET
import matplotlib.pyplot as plt
import numpy as np
import random
########################################## genetic algorithm #########################
class Evolution:
    def __init__(self, sizes,nb_individuals=100, nb_generations=500):
        self.nb_individuals = nb_individuals
        self.nb_generations = nb_generations
        self.neuro_size = sizes
        #population
        self.fitness = []
        self.population = []
        #save history for visulization
        self.fitness_history = []
        self.best_fitness_history = []

    def init_population(self):
        self.population = []
        self.fitness = [] #init fitness eachtime
        for i in range(self.nb_individuals):
            self.population.append(EVOLUNET(self.neuro_size))


    def calculate_fitness(self,training_data):
        self.fitness =[] #init fitness
        i_max = 0
        max_fitness = - np.inf
        for i,individual in enumerate(self.population):
            indi_fitness = individual.evaluate(training_data)
            #search for the max fitness individual
            if indi_fitness > max_fitness:
                i_max= i
                max_fitness = indi_fitness
            self.fitness.append(indi_fitness)
        return i_max,max_fitness

    def selection_tournament(self,tournament_size = 4):
        selected = []
        for _ in range(len(self.population)):
            tournament = random.sample(list(zip(self.population,self.fitness)), tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner)
        return selected

    def plot_fitness(self):
        fig,ax = plt.subplots()
        ax.plot(range(len(self.fitness_history)), self.fitness_history, label='Average fitness')
        ax.plot(range(len(self.best_fitness_history)), self.best_fitness_history, label = "Best fitness",linestyle = ":")
        ax.set_title(f"fitness along the {self.nb_generations} generations in a population of {self.nb_individuals} individuals ")
        ax.set(xlabel="Generation", ylabel=f'fitness')
        ax.legend()
        plt.show()

    def get_average_fitness(self):
        return sum(self.fitness) / len(self.fitness)

    def reproduce(self,selected,training_data):
        (parent,parent_fitness)= selected
        parent_weights = parent.weights
        parent_bias = parent.bias
        # create new child with mutated weights and bias
        child = EVOLUNET(self.neuro_size, parent_bias, parent_weights)
        child.mutate_param()
        child_fitness = child.evaluate(training_data)
        # return child, child_fitness
        if child_fitness > parent_fitness:
            return child,child_fitness
        else :
            return parent, parent_fitness

    def crossover(self, parent1, parent2):

        child1_weights = []
        child2_weights = []
        child1_bias = []
        child2_bias = []

        # Iterate through corresponding weights and biases of parent1 and parent2
        for w1, w2, b1, b2 in zip(parent1.weights, parent2.weights, parent1.bias, parent2.bias):
            # Convert lists to numpy arrays for element-wise operations
            w1, w2 = np.array(w1), np.array(w2)
            b1, b2 = np.array(b1), np.array(b2)
            alpha_w = np.random.uniform(size = w1.shape)
            alpha_b = np.random.uniform(size = b1.shape)
            # Perform crossover on weights and biases
            child1_weights.append(alpha_w * w1 + (1 - alpha_w) * w2)
            child2_weights.append(alpha_w * w2 + (1 - alpha_w) * w1)
            child1_bias.append(alpha_b * b1 + (1 - alpha_b) * b2)
            child2_bias.append(alpha_b * b2 + (1 - alpha_b) * b1)

        # Return weights and biases as lists of numpy arrays
        return child1_weights, child2_weights, child1_bias, child2_bias

    def generation_evolution(self,training_data, crossover=None):
        self.init_population()
        for generation in range(self.nb_generations):
            i_best_individual,best_fitness = self.calculate_fitness(training_data)
            best_individual = self.population[i_best_individual]
            average_fitness = self.get_average_fitness()



            #keep information for plot
            self.fitness_history.append(average_fitness)
            self.best_fitness_history.append(best_fitness)

            #selection par tournament
            sub_population = self.selection_tournament(tournament_size = 3)
            next_population = []
            if generation % 100 ==0:
                print("Generation {}".format(generation))
                # print(set(sub_population))
            if crossover:
                for i in range(0, len(sub_population), 2):
                    parent1 = self.population[i]
                    parent2 = self.population[i + 1]

                    child1_weights, child2_weights, child1_bias, child2_bias = self.crossover(parent1, parent2)

                    # create new population with children
                    child1 = EVOLUNET(self.neuro_size, child1_bias,child1_weights)
                    child2 = EVOLUNET(self.neuro_size, child2_bias,child2_weights)

                    # child1.mutate_param()
                    # child2.mutate_param()

                    child1_fitness = child1.evaluate(training_data)
                    child2_fitness = child2.evaluate(training_data)

                    if child1_fitness > parent1.fitness:
                        next_population.append(child1)
                    else:
                        next_population.append(parent1)
                    if child2_fitness > parent2.fitness:
                        next_population.append(child2)
                    else:
                        next_population.append(parent2)
            else:
                for (parent, parent_fitness) in sub_population:
                    next_indivi,next_indivi_fitness= self.reproduce((parent,parent_fitness),training_data)
                    next_population.append(next_indivi)
            self.population = next_population

        print("learning process finished")
        #return the network with best fitness

        i_optimized_individual, optimized_fitness = self.calculate_fitness(training_data)

        return self.population[i_optimized_individual]

