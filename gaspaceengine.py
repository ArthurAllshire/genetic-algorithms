import random

"""
The values

NULL: 00
Russian Engine: 01
Ion Engine: 10
Light Payloads: 11
"""

DESIRED_NUMBER = 59
POPULATION_SIZE = 100
CHROM_SIZE = 9
GENE_SIZE = 2
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.05
ILLEGAL_GENES = []

def run_program():
    current_generation = generate_population() 
    generations = 0
    solution = None
    while not solution:
        if len(current_generation[0]) == 1:
            solution = current_generation
        else:
            current_generation = next_generation(current_generation)
            print("G,", str(generations), ",F," , str(sum(current_generation[1])))
        generations += 1
    print("Solution Found!: " + solution[0][0])
    fitness(solution[0][0], print_info=True)

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
        if fitness_value == -1:
            return [[individual], [fitness_value]]
    return [next_generation, fitnesses]

def weighted_choice(weights):
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

def evaluate_individual(individual):
    chrom = []
    for x in range(1, CHROM_SIZE):
       chrom.append(int(individual[x*GENE_SIZE-GENE_SIZE:x*GENE_SIZE], 2))
    total = None
    operator = None
    money_left = 1000
    trip_time = 1600
    # russian ion payload
    used = [0, 0, 0]
    for gene in chrom:
        if gene == 1 and money_left >= 400 and used[0] < 3:
            if used[0] == 0:
                used[0] += 1
                money_left -= 400
                trip_time -= 200
            elif used[0] == 1:
                used[0] += 1
                money_left -= 400
                trip_time -= 100
        elif gene == 2 and money_left >= 150 and used[1] < 6:
            used[1] += 1
            money_left -= 150
            trip_time -= 50
        elif gene == 3 and money_left >= 50 and used[2] < 4:
            used[2] += 1
            money_left -= 50
            trip_time -= 25
    return money_left, trip_time, used 

def mutate(individual):
    new_gene = generate_gene()
    position = random.randint(1, CHROM_SIZE)
    return individual[:position*GENE_SIZE] + new_gene + individual[(position+1)*GENE_SIZE:] 

def crossover(individual_1, individual_2):
    crossover_position = random.randint(1, CHROM_SIZE)
    new_individual_1 = individual_1[0:crossover_position*GENE_SIZE]+individual_2[crossover_position*GENE_SIZE:]
    #new_individual_2 = individual_2[0:crossover_position*GENE_SIZE]+individual_1[crossover_position*GENE_SIZE:]
    return new_individual_1#, new_individual_2

def generate_individual():
    individual = ""
    for x in range(CHROM_SIZE):
        individual += generate_gene()
    return individual

def generate_gene():
    gene = ""
    for x in range(GENE_SIZE):
        gene += str(random.getrandbits(1))
    if gene not in ILLEGAL_GENES:
        return gene
    else:
        return generate_gene()

def fitness(individual, print_info=False):
    value = None
    money_left, trip_time, used = evaluate_individual(individual)
    value = 1/trip_time
    if print_info:
        print("Money Left: ", money_left)
        print("Trip Time: ", trip_time)
        print("\n Engines Used:\nRussian: ", used[0], "\nIon: ", used[1], "\nPayloads: ", used[2])
    if trip_time <= 1175:
        return -1
    return value

if __name__ == "__main__":
    run_program()
