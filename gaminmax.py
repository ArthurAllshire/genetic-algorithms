import random

"""
The values

8 bit integer
0-255
"""

DESIRED_NUMBER = 59
POPULATION_SIZE = 1000
CHROM_SIZE = 8
GENE_SIZE = 1
CROSSOVER_RATE = 0.2
MUTATION_RATE = 0.1
INPUT_STRUCTURE = [0, 50, 70, 120, 100, 80, 90, 20, 160, 40, 11, 1]
DESIRED_OUTPUT =  [0,  1,  1,   0,  0,  1,   1,  0,   0,  1,  0, 0]
# total number of 1s in the input structure
TOTAL_ON_BITS = sum(DESIRED_OUTPUT)
# total number of 0s in the input strucuture
TOTAL_OFF_BITS = len(DESIRED_OUTPUT) - sum(DESIRED_OUTPUT)
doubling_weights = [2**x for x in range(CHROM_SIZE)]

def run_program():
    current_generation = generate_population() 
    generations = 0
    solution = None
    while not solution:
        print("G," + str(generations) + ",F," + str(sum(current_generation[1])))
        if len(current_generation[0]) == 1:
            solution = current_generation
        else:
            current_generation = next_generation(current_generation)
        generations += 1
    print("Solution Found!: " + str(solution[0][0]))

def next_generation(generation):
    number = random.uniform(0.0, 1.0)
    next_generation = []
    while len(next_generation) < 50:
        next_selection = random.uniform(0.0, 1.0)
        if next_selection <= MUTATION_RATE:
            next_generation.append(mutate(generation[0][weighted_choice(generation[1])]))
        elif next_selection <= CROSSOVER_RATE + MUTATION_RATE:
            next_generation.append(crossover(generation[0][weighted_choice(generation[1])], generation[0][weighted_choice(generation[1])]))
        else:
            # copy an individual
            next_generation.append(generation[0][weighted_choice(generation[1])])
    fitnesses = []
    for individual in next_generation:
        fitness_value = fitness(individual)
        fitnesses.append(fitness_value)
        if fitness_value == 1:
            return [[individual], [fitness_value]]
    return [next_generation, fitnesses]

def weighted_choice(weights):
    if sum(weights) == 0.0:
        return random.randint(0,len(weights)-1)
    totals = []
    running_total = 0

    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random.random() * running_total
    for i, total in enumerate(totals):
        if rnd < total:
            return i

def generate_population():
    population = []
    fitnesses = []
    for x in range(POPULATION_SIZE):
        individual = generate_individual()
        fitness_value = fitness(individual)
        population.append(individual)
        fitnesses.append(fitness_value)
        if fitness_value == -1:
            return [[individual], [fitnesses]]
    return [population, fitnesses]

def evaluate_chrom(chrom):
    return int(chrom, 2)

def evaluate_individual(individual):
    min_value = evaluate_chrom(individual[0])
    max_value = evaluate_chrom(individual[1])
    if min_value > max_value:
        return None, None
    output = []
    for number in INPUT_STRUCTURE:
        output.append(int(min_value<=number<=max_value))
    true_positives = 0
    true_negatives = 0
    for output_number, correct_output in zip(output, DESIRED_OUTPUT):
        if output_number and correct_output:
            true_positives += 1
        if not output_number and not correct_output:
            true_negatives += 1
    return true_positives, true_negatives

def mutate(individual):
    new_gene = generate_gene()
    # be more likely to choose less significant bits in the chromesome
    position = weighted_choice(doubling_weights)
    mutate_chrom = random.randint(0, 1)
    individual[mutate_chrom] = individual[mutate_chrom][:position*GENE_SIZE] + new_gene + individual[mutate_chrom][(position+1)*GENE_SIZE:] 
    return individual

def crossover(individual_1, individual_2):
    crossover_position = random.randint(1, CHROM_SIZE)
    crossover_chrom = random.randint(0,1)
    new_individual_1 = individual_1
    new_individual_1[crossover_chrom] = individual_1[crossover_chrom][0:crossover_position*GENE_SIZE]+individual_2[crossover_chrom][crossover_position*GENE_SIZE:]
    #new_individual_2 = individual_2[0:crossover_position*GENE_SIZE]+individual_1[crossover_position*GENE_SIZE:]
    return new_individual_1#, new_individual_2

def generate_chrom():
    individual = ""
    for x in range(CHROM_SIZE):
        individual += generate_gene()
    return individual

def generate_individual():
    return [generate_chrom(), generate_chrom()]

def generate_gene():
    gene = ""
    for x in range(GENE_SIZE):
        gene += str(random.getrandbits(1))
    return gene

def fitness(individual):
    value = None
    true_positives, true_negatives = evaluate_individual(individual)
    if true_positives == None:
        return 0.0
    value = (true_positives/TOTAL_ON_BITS)*(true_negatives/TOTAL_OFF_BITS)
    return value

if __name__ == "__main__":
    run_program()
