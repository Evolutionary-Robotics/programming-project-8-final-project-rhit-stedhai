import Simulation
import Neural_Network
import CTRNN
import Evolutionary_Algorithm as ea
import numpy as np
import matplotlib.pyplot as pyplot

simLength = 200
numSensors = 2
size = 4
numGenes = size*3
weightsRange = 10
biasRange = 10

numGenerations = 300
recombinationRate = 0.5
mutationRate = 0.1
elitism = 5
populationSize = 50

def fintnessFunction(genotype):
    preyNeuralNet = CTRNN.CTRNN(size=size)
    preyNeuralNet.intialize_from_genetic_algorithm(genotype)
    sim = Simulation.Simulation(initialPreyPopSize=5, simLength=simLength, preyNeuralNet= preyNeuralNet, preyVelocity = 3)
    preyPop, preyEaten, numPreyBred, numPreyDied = sim.run(plot = False)
    
    if preyPop == 0:
        return 0
    elif preyPop < 5:
        return (numPreyBred+preyEaten)/(5-preyPop)
    else:
        return numPreyBred+preyEaten
    
def mutationFunction(genotype, mutationRate):
    for g in range(len(genotype)):
        if np.random.random() < mutationRate:
            if np.random.random() > 0.5:
                genotype[g] += np.random.random()*weightsRange
            else:
                genotype[g] -= np.random.random()*weightsRange
    
    return genotype


populationOne = []
for p in range(populationSize):
    individual = [0]*numGenes
    for g in range(numGenes):
        individual[g] = np.random.random()
        if np.random.random() > 0.5:
            individual[g] *= -1
    individual = individual
    populationOne.append(individual)
ea1 = ea.Evolutionary_Algorithm(populationSize=populationSize, numGenerations=numGenerations, 
                               recombinationRate=recombinationRate, mutationRate=mutationRate, 
                               elitism=elitism, numGenes=numGenes, population=populationOne,
                               fitness_function=fintnessFunction, mutation_function=mutationFunction)
best, average, bestOveralls, bestIndvidual, generations, elitePopulation = ea1.run()
stepsList = [0]*generations
bestScore = [0]*generations
averageScores = [0]*generations
bestOverallScores = [0]*generations
for g in range(generations):
    stepsList[g] = g
    bestScore[g] = best[g]
    averageScores[g] = average[g]
    bestOverallScores[g] = bestOveralls[g]

print("best Score: ", bestScore[-1])
print("best: ", bestIndvidual)
print()
figure, axes = pyplot.subplots()
axes.plot(stepsList, bestScore)
axes.plot(stepsList, averageScores)
axes.plot(stepsList, bestOverallScores)
pyplot.xlabel("Generation Number")
pyplot.ylabel("Scores")
pyplot.show()

preyCount = 1
for ep in elitePopulation:
    print("Best Prey ", preyCount)
    print(ep[2])
    print()
    preyCount += 1
    preyNeuralNet = CTRNN.CTRNN(size=size)
    preyNeuralNet.intialize_from_genetic_algorithm(ep)
    sim = Simulation.Simulation(initialPreyPopSize=5, simLength=simLength, preyNeuralNet= preyNeuralNet, preyVelocity=3)
    populationAfter = sim.run(plot = True)


# preyPop = []
# preyPop.append([-24.464772335970267, -23.285715433806892, 53.910365347007286, -0.6344601827347347, 1.0671412760322156, 0.5722588673452887, 25.214476910223823, -60.6422030114087, 0.4569949153650902, 0.4485957340070029, -26.887620124952843, 8.598859235979376])
# preyPop.append([-1.5013565786984637, -25.073658107669537, 39.07405552304383, -6.666325055921666, 1.0671412760322156, 0.5722588673452887, 21.560399645527404, -67.1781991988093, 0.4569949153650902, 0.4485957340070029, -28.98645695437562, 1.5405875469503822])
# preyPop.append([-35.99556332309379, -5.587085722775768, 30.907564847247677, 41.8954631986568, -0.17460563339932778, -0.595326004588772, 8.377062538433727, 16.648997064700755, 0.4140235136112066, 0.4000515758845532, -20.542886108687192, -21.534725021945487])
# preyPop.append([-44.990996604840255, -19.96044189966404, 45.70173240928969, 42.7088754418754, -0.17460563339932778, -0.595326004588772, 13.589818964738825, 19.916538848656792, 0.4140235136112066, 0.4000515758845532, -11.362628456625824, -31.52302140898508])
# preyPop.append([-23.258096945997142, -7.064989483189624, 14.880222348104143, 74.77969725249625, -1.346030818276283, -0.8161533354821758, -23.08422886292335, 35.49865858629866, 0.42228822375136754, 0.4797347141228825, -1.1256990538224678, -9.899522247928495])
# preyPop.append([-29.81153217223877, -14.052483679017879, -9.635011642049715, 75.17627989353912, -1.346030818276283, -0.8161533354821758, -22.811175383896796, 57.76553874394855, 0.42228822375136754, 0.4797347141228825, 13.506248165551458, -14.207449234965983])
# preyNeuralNet = CTRNN.CTRNN(size=size)
# preyNeuralNet.intialize_from_genetic_algorithm(preyPop[5])
# sim = Simulation.Simulation(initialPreyPopSize=5, simLength=simLength, preyNeuralNet= preyNeuralNet, preyVelocity = 3)
# sim.run(plot = True)
