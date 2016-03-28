import random

"""
The values


0: 0000
1: 0001
2: 0010
3: 0011
4: 0100
5: 0101
6: 0110
7: 0111
8: 1000
9: 1001
+: 1010
-: 1011
*: 1100
/: 1101
"""

DESIRED_NUMBER = 59
POPULATION_SIZE = 100
CHROM_SIZE = 9
GENE_SIZE = 4
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.01
ILLEGAL_GENES = ["1110", "1101"]

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
    print("Solution Found!: " + solution[0][0])

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
       chrom.append(int(individual[x*GENE_SIZE-4:x*GENE_SIZE], 2))
    total = None
    operator = None
    for gene in chrom:
        if gene < 10:
            # it is a number
            if total == None:
                total = gene
            elif total != None and operator != None:
                if operator == 10:
                    # add
                    total += gene
                elif operator == 11:
                    # subtract
                    total -= gene
                elif operator == 12:
                    # multiply
                    total *= gene
                else:
                    # divide
                    try:
                        total /= gene
                    except ZeroDivisionError:
                        pass
                operator = None
        elif gene > 9:
            if operator == None:
                operator = gene
    return total

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

def fitness(individual):
    value = None
    output = evaluate_individual(individual)
    try:
        value = 1/abs(DESIRED_NUMBER-output)
    except ZeroDivisionError:
        return -1
    return value

if __name__ == "__main__":
    run_program()
