import numpy as np

class Evolutionary_Algorithm:
    def __init__(self, populationSize, numGenerations, numGenes, recombinationRate, mutationRate, elitism, population, fitness_function, mutation_function):
        self.populationSize = populationSize
        self.numGenerations = numGenerations
        self.numGenes = numGenes
        self.recombinationRate = recombinationRate
        self.mutationRate = mutationRate
        self.fitness_function = fitness_function
        self.mutate_function = mutation_function
        self.elitism = elitism
        self.population = population

    def run(self):
        bestScores = [0] * self.numGenerations
        averageScores = [0] * self.numGenerations
        bestOveralls = [0] * self.numGenerations
        bestOverall = self.fitness_function(self.population[0])
        bestIndividualOverall = self.population[0]
        elitePopulation = []
        for generation in range(self.numGenerations):
            localPopulation = self.population[:]
            self.fitnessScores = [0]*self.populationSize
            for p in range(len(self.population)):
                self.fitnessScores[p] = self.fitness_function(self.population[p])
                # if self.fitnessScores[p] > 25:
                #     print("above 25: ", self.population[p][4])

            self.quick_sort_population_by_fitness(startNum=0, endNum=len(self.population)-1)

            elitePopulation = []
            for e in range(self.elitism):
                elitePopulation.append(self.population[e])
            
            self.population = self.generate_new_population(elitePopulation=elitePopulation,localPopulation=localPopulation)

            best = -10
            average = 0
            bestIndividual = None
            for p in range(len(self.population)):
                fitness = self.fitnessScores[p]
                if fitness >= best:
                    best = fitness
                    bestIndividual = self.population[p]
                if fitness >= bestOverall:
                    bestOverall = fitness
                    bestIndividualOverall = self.population[p]
                
                average += fitness

            if generation % 10 == 0:
                print("generation: ", generation)
                print("average: ", average/self.populationSize)
                print("best fitness: ", best)
                print("best overall fitness: ", bestOverall)
                #print("best individual: ", bestIndividual)
                #print("best overal agent: ", bestIndividualOverall)
                print()

            bestScores[generation] = best
            averageScores[generation] = average/self.populationSize
            bestOveralls[generation] = bestOverall

        return bestScores, averageScores, bestOveralls, bestIndividualOverall, generation, elitePopulation
                

    def generate_new_population(self, elitePopulation, localPopulation):
        new_population = [0] * self.populationSize
        pos = 0

        if self.elitism > 0:
            for e in elitePopulation:
                new_population[pos] = e
                pos += 1

        for current in range(pos, self.populationSize):
            parentOne, parentTwo = self.get_parents(localPopulation=localPopulation)
            child = self.create_child(parentOne, parentTwo)
            if child == None:
                print(parentOne)
                print(parentTwo)
                print()
            new_population[current] = child

        return new_population

    
    def create_child(self, parentOne, parentTwo):
        child = [0]*self.numGenes
        for g in range(self.numGenes):
            if np.random.random() < self.recombinationRate:
                child[g] = parentOne[g]
            else:
                child[g] = parentTwo[g]
        child = self.mutate_function(child, self.mutationRate)
        return child

    def get_parents(self, localPopulation):
        parents = [0, 0]
        totalScore = 0
        for i in range(self.populationSize):
            totalScore += (i+1)
 
        
        for j in range(2):
            rand = np.random.random()
            score = 0
            for i in range(self.populationSize):
                score += self.populationSize - i
                if rand < score/totalScore:
                    parents[j] = self.population[i]
                    break
                
        return parents[0], parents[1]

    def quick_sort_population_by_fitness(self, startNum, endNum):
        if startNum < endNum:
            paritionNum = self.partition(startNum, endNum)
            self.quick_sort_population_by_fitness(startNum=startNum, endNum=paritionNum-1)
            self.quick_sort_population_by_fitness(startNum=paritionNum+1, endNum=endNum)

    def partition(self, startNum, endNum):
        pivot = self.fitnessScores[endNum]
        current = startNum - 1
        for i in range(startNum, endNum):
            if self.fitnessScores[i] > pivot:
                current += 1
                temp = self.population[current]
                tempScore = self.fitnessScores[current]
                self.population[current] = self.population[i]
                self.fitnessScores[current] = self.fitnessScores[i]
                self.population[i] = temp
                self.fitnessScores[i] = tempScore

        current += 1
        temp = self.population[current]
        self.population[current] = self.population[endNum]
        self.population[endNum] = temp

        return current

        