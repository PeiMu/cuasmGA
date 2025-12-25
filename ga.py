from collections import Counter
import random
from typing import Callable, Optional
from sass_kernel import SassKernel
from sassgen import extract_kernel_sass, write_sass_file


POP_SIZE = 10   #population size
MUTATION_RATE = 0.1
NUM_GENERATIONS = 10
ELITE_SIZE = 4

class Individual:
    def __init__(self, kernel_section):
        self.sass = kernel_section  # List of instructions (strings)
        self.fitness: Optional[float] = None


class GeneticAlgorithm:
    original_kernel_section: Optional[list] = None

    def __init__(self, kernel_section,test_correctness,test_performance):
        self.original_kernel_section = kernel_section
        self.counter = Counter(kernel_section)
        # Check for any duplicates
        has_duplicate = any(count > 1 for count in self.counter.values())
        print(f"Existing duplicate？：{has_duplicate}")
        self.test_correctness: Callable[[Individual], bool] = test_correctness
        self.test_performance: Callable[[Individual], float]= test_performance
    def evaluate_fitness(self,individual) -> float:
        fitness = self.test_performance(individual)
        return fitness
        # cubin = write_sass_file(individual.sass)  # write back to cubin
        # run_benchmark(cubin)
    # def run_benchmark(cubin):



    def initialize_population(self,original_kernel_section): 
        # Initialize population
        population = []
        for _ in range(POP_SIZE):
            shuffled = random.sample(original_kernel_section, len(original_kernel_section))
            if Counter(shuffled) != self.counter:
                print(f"counter false")
            population.append(Individual(shuffled))
        return population

    def tournament_selection(self,population, k = 4):  #to reduce the noise, try rank-based selection later
        contenders = random.sample(population, k)
        best = min(contenders, key=lambda x: x.fitness)
        return best

    def crossover(self,parent1 : Individual, parent2 : Individual):
        #add dependency here for later optimization

        size = len(parent1.sass)
        #create selection point
        point1 = random.randint(0, size-2)
        point2 = random.randint(point1+1, size-1)
        #create child1
        child1_sass = [None] * size
        child1_sass[point1:point2] = parent1.sass[point1:point2]
        remainingp2 = [x for x in parent2.sass if x not in child1_sass[point1:point2]]
        fill_idx = 0
        for i in range(size):
            if child1_sass[i] is None:
                child1_sass[i] = remainingp2[fill_idx]
                fill_idx += 1
        if fill_idx > len(remainingp2): print("error: fill_idx = len(remainingp1)")
        #create child2
        child2_sass = [None] * size
        child2_sass[point1:point2] = parent2.sass[point1:point2]
        remainingp1 = [x for x in parent1.sass if x not in child2_sass[point1:point2]]
        fill_idx = 0
        for i in range(size):
            if child2_sass[i] is None:
                child2_sass[i] = remainingp1[fill_idx]
                fill_idx += 1
        if fill_idx > len(remainingp1): print("error: fill_idx = len(remainingp1)")
        if Counter(child1_sass) == self.counter:
            child1 = Individual(child1_sass)
        # child1,child2 = Individual(child1_sass),Individual(child2_sass)
            child1.fitness = self.evaluate_fitness(child1)
        else: child1 = None

        if Counter(child2_sass) == self.counter:
            child2 = Individual(child2_sass)
            child2.fitness = self.evaluate_fitness(child2)
        else: child2 = None
        return child1,child2


    def mutate(self,individual) -> Individual:
        if(random.random() < MUTATION_RATE):
            i, j = random.sample(range(len(individual.sass)),2)
            individual.sass[i], individual.sass[j] = individual.sass[j], individual.sass[i]
            individual.fitness = self.evaluate_fitness(individual)
        #evaluate the mutated individual
        return individual

    def run_ga(self, originol_pure_kernel:list):
        population = self.initialize_population(originol_pure_kernel)
        #get population fitness
        for individual in population:
            individual.fitness = self.evaluate_fitness(individual)
        
        for gen in range(NUM_GENERATIONS):
            print(f"GEN {gen} best fitness: {min(individual.fitness for individual in population)} ")

            #keep the elite individual
            population.sort(key = lambda x: x.fitness)
            next_gen = population[:ELITE_SIZE]

            while(len(next_gen) < POP_SIZE):
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                child1,child2 = self.crossover(parent1, parent2)
                if child1 is not None:
                    next_gen.append(self.mutate(child1))
                if child2 is not None:
                    next_gen.append(self.mutate(child2))
            
            population = next_gen
        
        best = min(population, key = lambda x: x.fitness)
        
        
        return best
            




# if __name__ == '__main__':
#     sass,kernel_section = extract_kernel_sass("cuasmrl_kernel_0d1d2_0.pkl")
#     pure_kernel = SassKernel(sass,kernel_section)._get_kernel()
#     print(len(pure_kernel))


