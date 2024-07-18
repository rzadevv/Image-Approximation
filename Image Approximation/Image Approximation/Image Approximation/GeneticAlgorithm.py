import random
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt

class GeneticSettings:
    # Get the image name from user input
    image_name = input('Enter the image name: ')

    def __init__(self):
        # Genetic algorithm settings
        self.GENE_COUNT = 400 
        self.MUT_PROB = 0.002
        self.POP_SIZE = 600
        self.IDEAL_FITNESS = 0
        self.ELITE_COUNT = 20
        self.FITNESS_TYPE = "min"
        self.CROSS_PROB = 0.95
        self.GENE_TYPE = "binary"

class Chromosome:
    def __init__(self, config):
        self.config = config
        # Generate random genes for the chromosome
        self.genes = [random.randint(0, 1) for _ in range(self.config.GENE_COUNT)]
        self.fitness = 0
        self.original_fitness = 0.0
        self.normalized_fitness = 0.0
        self.end_range = 0.0
    
    def mutate(self):
        # Mutate the genes based on mutation probability
        for i in range(len(self.genes)):
            if random.uniform(0, 1) <= self.config.MUT_PROB:
                self.genes[i] = self.flip_gene(self.genes[i])
    
    def flip_gene(self, gene_value):
        # Flip the gene value
        return 1 if gene_value == 0 else 0

class RouletteWheelSelection:
    def __init__(self, chromosomes, config):
        self.chromosomes = chromosomes
        self.config = config
        self.create_cumulative_probabilities()
    
    def select_parents(self):
        # Roulette wheel selection to select parents
        random1 = random.uniform(0, 1)
        chromosome1 = None
        for chromosome in self.chromosomes:
            if random1 <= chromosome.end_range:
                chromosome1 = Chromosome(self.config)
                chromosome1.genes = chromosome.genes
                break
        
        chromosome2 = None
        while True:
            random2 = random.uniform(0, 1)
            for chromosome in self.chromosomes:
                if random2 <= chromosome.end_range:
                    chromosome2 = Chromosome(self.config)
                    chromosome2.genes = chromosome.genes
                    
                    if chromosome1.genes != chromosome2.genes:
                        return chromosome1, chromosome2

    def create_cumulative_probabilities(self):
        # Create cumulative probabilities for roulette wheel selection
        self.calculate_cumulative_sum()
        self.get_normalized_fitness()
        current_sum = 0
        for chromosome in self.chromosomes:
            current_sum += chromosome.normalized_fitness
            chromosome.end_range = current_sum
    
    def calculate_cumulative_sum(self):
        # Calculate cumulative sum of fitness values
        cum_sum = sum(chromosome.fitness for chromosome in self.chromosomes)
        self.cum_sum = cum_sum
    
    def get_normalized_fitness(self):
        # Calculate normalized fitness for each chromosome
        for chromosome in self.chromosomes:
            chromosome.normalized_fitness = chromosome.fitness / self.cum_sum

class GeneticAlgorithm:
    def __init__(self):
        self.config = GeneticSettings()
        self.GEN_COUNT = 0
        self.selected_chromosomes = {}
        self.best_fitnesses = []  

    def crossover(self, chromosome1, chromosome2):
        # Perform crossover between two chromosomes
        crossover_point = random.randint(1, len(chromosome1.genes) - 1)
        child1_genes = chromosome1.genes[:crossover_point] + chromosome2.genes[crossover_point:]
        child2_genes = chromosome2.genes[:crossover_point] + chromosome1.genes[crossover_point:]
        chromosome_to_consider1 = Chromosome(self.config)
        chromosome_to_consider2 = Chromosome(self.config)

        if random.uniform(0, 1) <= self.config.CROSS_PROB:
            chromosome_to_consider1.genes = child1_genes
        else:
            chromosome_to_consider1.genes = chromosome1.genes.copy()
        
        if random.uniform(0, 1) <= self.config.CROSS_PROB:
            chromosome_to_consider2.genes = child2_genes
        else:
            chromosome_to_consider2.genes = chromosome2.genes.copy()

        return chromosome_to_consider1, chromosome_to_consider2
    
    def shift_if_negative_fitness(self, chromosomes):
        # Shift fitness values if negative fitness is encountered
        min_fitness = min(chromosome.fitness for chromosome in chromosomes)

        if min_fitness >= 0:
            return
        
        for chromosome in chromosomes:
            chromosome.fitness += min_fitness * -1
    
    def evolve(self, num_generations, fitness_function, step_execution=None):
        chromosomes = [Chromosome(self.config) for _ in range(self.config.POP_SIZE)]
        best_individual = Chromosome(self.config)
        best_individual.fitness = -1

        for generation in range(num_generations):
            self.selected_chromosomes = {} 

            for chromosome in chromosomes:
                fitness_value = fitness_function(chromosome)
                chromosome.original_fitness = fitness_value

                if self.config.FITNESS_TYPE == 'min':
                    fitness_value *= -1
                chromosome.fitness = fitness_value
            
            self.shift_if_negative_fitness(chromosomes)

            rw = RouletteWheelSelection(chromosomes, self.config)
            next_gen_chromosomes = []

            for _ in range(self.config.POP_SIZE // 2):
                chromosome1, chromosome2 = rw.select_parents()
                chromosome1, chromosome2 = self.crossover(chromosome1, chromosome2)
                chromosome1.mutate()
                chromosome2.mutate()

                if str(chromosome1.genes) not in self.selected_chromosomes:
                    next_gen_chromosomes.append(chromosome1)
                    self.selected_chromosomes[str(chromosome1.genes)] = 1

                if str(chromosome2.genes) not in self.selected_chromosomes:
                    next_gen_chromosomes.append(chromosome2)
                    self.selected_chromosomes[str(chromosome2.genes)] = 1
                        
            chromosomes.sort(key=lambda x: x.fitness, reverse=True)
            
            best_individual = copy.deepcopy(chromosomes[0])
            self.best_fitnesses.append(best_individual.original_fitness)
            
            if best_individual.original_fitness == self.config.IDEAL_FITNESS:
                return best_individual
            
            for i in range(self.config.ELITE_COUNT):
                if str(chromosomes[i].genes) not in self.selected_chromosomes:
                    next_gen_chromosomes[len(next_gen_chromosomes) - i - 1] = copy.deepcopy(chromosomes[i])
            
            if step_execution is not None:
                step_execution(generation_number=self.GEN_COUNT, best_individual=chromosomes[0])

            self.GEN_COUNT += 1
            print("Best => " + str(chromosomes[0].original_fitness) +"")
            chromosomes = next_gen_chromosomes
        return best_individual

def fitness_function(chromosome):
    # Fitness function: Calculate fitness based on the difference between chromosome genes and image pixels
    fitness_value = 0
    for i in range(len(image_flattened)):
        fitness_value += pow((image_flattened[i] - chromosome.genes[i]), 2)
    return fitness_value    

image = cv2.imread(GeneticSettings.image_name, 0)
image_flattened = image.flatten().tolist()

for i in range(len(image_flattened)):
    if image_flattened[i] == 255:
        image_flattened[i] = 1
    else:
        image_flattened[i] = 0

image = np.reshape(image_flattened, (20, 20)) * 255.0

def step_executor(generation_number, best_individual):
    # Step executor: Display original and formed images
    best_individual = np.array(best_individual.genes)
    best_individual = best_individual * 255.0
    best_individual = np.reshape(best_individual, (20, 20))
    best_individual = cv2.resize(best_individual, (200, 200))
    cv2.imshow("ORIGINAL", cv2.resize(image, (200, 200)))
    cv2.imshow("FORMED", best_individual)
    key = cv2.waitKey(1)

util = GeneticAlgorithm()
node = util.evolve(300, fitness_function, step_execution=step_executor)

# Plotting the fitness evolution
plt.plot(range(1, len(util.best_fitnesses) + 1), util.best_fitnesses)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Evolution of Best Fitness')
plt.grid(True)
plt.show()
