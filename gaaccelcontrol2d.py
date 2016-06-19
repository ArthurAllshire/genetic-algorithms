import random
import numpy as np

"""
The values

-10: 00000 - 0
  0: 01011 - 10
 10: 10100 - 20
  +: 10101 - 21
  -: 10110 - 22
  *: 10111 - 23
  /: 11000 - 24
 **: 11001 - 25
current_velocity: 11010 - 26
current_distance_to_target: 11011 - 27
"""

DESIRED_NUMBER = 59
POPULATION_SIZE = 10000
CHROM_SIZE = 20 # genes
GENE_SIZE = 5 # bits
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.1
ILLEGAL_GENES = ["11001", "11100", "11101", "11110", "11111"]
FITNESS_THRESHOLD = 1/20 * 10

def run_program():
    current_generation = generate_population() 
    generations = 0
    solution = None
    while not solution:
        print("G," + str(generations) + ",F," + str(sum(current_generation[1])))
        print("Fittest: ", current_generation[0][np.argmax(current_generation[1])])
        if len(current_generation[0]) == 1:
            solution = current_generation
        else:
            current_generation = next_generation(current_generation)
        generations += 1
    print("Solution Found!: " + solution[0][0])
    print("Result:")
    fitness_value = fitness(solution[0][0], print_dist = True)


def next_generation(generation):
    #mutation_rate = MUTATION_RATE * 1.5**1/((sum(generation[1])/len(generation[1]))/FITNESS_THRESHOLD)
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

def evaluate_individual(chrom, velocity, distance, target, print_functions=False):
    total = None
    operator = None
    for gene in chrom:
        if 0 <= gene <= 20 or 26 <= gene <= 27:
            if gene == 26:
                gene = velocity
            elif gene == 27:
                gene = target - distance
            else:
                gene -= 10
                gene /- 10
            # it is a number
            if total == None:
                total = gene
            elif total != None and operator != None:
                if operator == 21:
                    # add
                    total += gene
                elif operator == 22:
                    # subtract
                    total -= gene
                elif operator == 23:
                    # multiply
                    total *= gene
                elif operator == 24:
                    # divide
                    try:
                        total /= gene
                    except ZeroDivisionError:
                        pass
                elif operator == 25:
                    total **= gene
                operator = None
        elif 20 < gene < 26:
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

def fitness(individual, print_dist=False):
    chrom = []
    for x in range(1, CHROM_SIZE):
       chrom.append(int(individual[x*GENE_SIZE-GENE_SIZE:x*GENE_SIZE], 2))
       #print(int(individual[x*GENE_SIZE-GENE_SIZE:x*GENE_SIZE], 2))
    smallest_dist = 10
    times_until_settled = []
    fitnesses = []
    overshoot = False
    for dist_x in range(3):
        for dist_y in range(3):
            velocity_x = 0
            distance_x = 0
            velocity_y = 0
            distance_y = 0
            since_target = 0
            target_x = smallest_dist * 10**(dist_x)
            target_y = smallest_dist * 10**(dist_y)
            time_until_settled = 0
            correct = True
            while since_target < 2 and time_until_settled < 100:
                if print_dist:
                    print("TIME PERIODS: ", time_until_settled, " DISTANCEX: ", distance_x, " TARGETX: ", target_x,
                            " DISTANCEY: ", distance_y, " TARGETY: ", target_y)
                output_x = evaluate_individual(chrom, velocity_x, distance_x, target_x, print_dist)
                output_y = evaluate_individual(chrom, velocity_y, distance_y, target_y, print_dist)
                if output_x >= 10:
                    output_x = 10
                elif output_x <= -10:
                    output_x = -10
                if output_y >= 10:
                    output_y = 10
                elif output_y <= -10:
                    output_y = -10
                velocity_x += output_x
                distance_x += velocity_x
                velocity_y += output_y
                distance_y += velocity_y
                if distance_x > target_x or distance_y > target_y:
                    overshoot = True
                #distance += output
                if abs(distance_x - target_x) <= 2 and abs(distance_y - target_y) <= 2:
                    since_target += 1
                else:
                    since_target = 0
                #print(since_target)
                time_until_settled += 1
                if time_until_settled > 10 and (distance_x == 0 or distance_y == 0) or distance_x > target_x * 10 or distance_y > target_y * 10 or distance_x < 0 or distance_y < 0:
                    break
            times_until_settled.append(time_until_settled)
            value = 0
            if since_target == 0 and not (abs(distance_x - target_x) <= 2) and not (abs(distance_y - target_y) <= 2):
                correct = False
                if  0 < distance_x > target_x and 0 < distance_y > target_y:
                    if print_dist:
                        print("TARGETX: ", target_x, " FIRST TRIPPED")
                        print("TARGETY: ", target_y, " FIRST TRIPPED")
                    value = 1/((abs(target_x - distance_x)+abs(target_y-distance_y))*10000)
                elif 0 < distance_x < target_x and 0< distance_y < target_y:
                    if print_dist:
                        print("TARGETX: ", target_x, " SECOND TRIPPED")
                        print("TARGETY: ", target_y, " SECOND TRIPPED")
                    value = 1/((abs(target_x - distance_x)+abs(target_y-distance_y))*1000)
                else:
                    if print_dist:
                        print("TARGETX: ", target_x, " THIRD TRIPPED")
                        print("TARGETY: ", target_y, " THIRD TRIPPED")
                    value = 1/10000000000000000
            else:
                if print_dist:
                    print("TARGETX: ", target_x, " TARGETY ", target_y, "REACHED TARGET")
                value = 1/time_until_settled
            value *= (target_x**2 + target_y**2)**0.5/10.0
            fitnesses.append(value)
    value = sum(fitnesses)/len(fitnesses)
    if not correct:
        value /= 2
    if overshoot:
        value *= 1/3
    if value > FITNESS_THRESHOLD and correct:
        return 1.0
    return value

if __name__ == "__main__":
    run_program()
