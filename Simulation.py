import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import Agent

class Simulation:
    def __init__(self, initialPreyPopSize, simLength, preyNeuralNet, preyVelocity):
        self.preyPopSize = initialPreyPopSize
        self.simLength = simLength
        self.preyNeuralNet = preyNeuralNet
        self.preyVelocity=preyVelocity
        self.preyPop = self.initializePopulation()

    def initializePopulation(self):
        preyPopulation = []
        # use genotypes here
        for p in range(self.preyPopSize):
            agent = Agent.Agent(xpos = np.random.randint(-50,50), ypos = np.random.randint(-50,50), 
                                velocity = self.preyVelocity, angle = np.random.random()*np.pi*2, numSensors = 2, sensorAngle=35)
            preyPopulation.append(agent)

        return preyPopulation

    def run(self, plot):
        thingsPreyEat = 0.0
        numPreyBred = 0.0
        numPreyDied = 0
        if plot:
            figure, axes = plt.subplots()
            axes.set_xlim(left=-90,right=90)
            axes.set_ylim(bottom=-90,top=90)

        grassLocations = []
        for step in range(self.simLength):
            if len(self.preyPop) == 0:
                break

            if step % 10 == 0:
                randX = np.random.randint(-80, 80)
                randY = np.random.randint(-80, 80)
                grassLocations.append([randX, randY])
            
            # update prey
            for prey in self.preyPop:
                # Prey dies
                if prey.getHunger() < 0:
                    self.preyPop.remove(prey)
                    numPreyDied += 1
                # Prey is hungry so its looking for food
                else: 
                    target = None
                    if len(grassLocations) > 0:
                        target = grassLocations[0]
                    for grassPatch in grassLocations:
                        distanceDifference = np.sqrt((prey.getXPos()-grassPatch[0])**2 + (prey.getYPos()-grassPatch[1])**2) - np.sqrt((prey.getXPos()-target[0])**2 + (prey.getYPos()-target[1])**2)
                        if distanceDifference < 0:
                            target = grassPatch

                    targetMate = self.preyPop[0]
                    if self.preyPop[0] is prey:
                        if len(self.preyPop) < 2:
                            targetMate = None
                        else:
                            targetMate = self.preyPop[1]
                    for mate in self.preyPop:
                        if mate is not prey:
                            distanceDifference = np.sqrt((prey.getXPos()-mate.getXPos())**2 + (prey.getYPos()-mate.getYPos())**2) - np.sqrt((prey.getXPos()-targetMate.getXPos())**2 + (prey.getYPos()-targetMate.getYPos())**2)
                            if distanceDifference < 0:
                                targetMate = mate
                    
                    prey.walk(walkType = 'Prey Neural Net', target = target, targetMate = targetMate, neuralNet = self.preyNeuralNet)
                    if target is not None:
                        curDistance = np.sqrt((prey.getXPos()-target[0])**2 + (prey.getYPos()-target[1])**2)
                        previousDistance = np.sqrt((prey.getPreviousXPos()-target[0])**2 + (prey.getPreviousYPos()-target[1])**2)
                        if curDistance < previousDistance:
                            thingsPreyEat += 0.1
                        if np.abs(target[0] - prey.getXPos()) < 6 and np.abs(target[1] - prey.getYPos()) < 6:
                            # Found grass to eat
                            prey.eat()
                            grassLocations.remove(target)
                            thingsPreyEat += 1
                    if targetMate is not None:
                        if not prey.getHasBred():
                            curDistance = np.sqrt((prey.getXPos()-targetMate.getXPos())**2 + (prey.getYPos()-targetMate.getYPos())**2)
                            previousDistance = np.sqrt((prey.getPreviousXPos()-targetMate.getPreviousXPos())**2 + (prey.getPreviousYPos()-targetMate.getPreviousYPos())**2)
                            if curDistance < previousDistance:
                                numPreyBred += 0.01
                            if not targetMate.getHasBred():
                                if np.abs(targetMate.getXPos() - prey.getXPos()) < 4 and np.abs(targetMate.getYPos() - prey.getYPos()) < 4:
                                    # Found a mate
                                    targetMate.breed()
                                    prey.breed()
                                    newPrey = Agent.Agent(xpos = prey.getXPos(), ypos = prey.getYPos(), 
                                                        velocity = self.preyVelocity, angle = np.random.random()*np.pi*2, numSensors = 2, sensorAngle=35)
                                    self.preyPop.append(newPrey)
                                    numPreyBred += 2
                    
                if plot:
                    # if target is not None:
                    #     axes.plot([prey.getXPos(), prey.getYPos()], [target[0], target[1]])
                    if prey.getHunger() < 40:
                        axes.plot(prey.getXPos(), prey.getYPos(), 'o', markersize = 6, color = 'teal')
                        sensors = prey.getSensors()
                        axes.plot(sensors[0][0], sensors[0][1], 'o', markersize = 1, color = 'black')
                        axes.plot(sensors[1][0], sensors[1][1], 'o', markersize = 1, color = 'black')
                    else:
                        axes.plot(prey.getXPos(), prey.getYPos(), 'o', markersize = 6, color = 'blue')
                        sensors = prey.getSensors()
                        axes.plot(sensors[0][0], sensors[0][1], 'o', markersize = 1, color = 'black')
                        axes.plot(sensors[1][0], sensors[1][1], 'o', markersize = 1, color = 'black')
            # update grass
            if plot:
                for grassPatch in grassLocations:
                    grassCircle = plt.Circle((grassPatch[0], grassPatch[1]), 5, color='green', fill=True)
                    axes.add_artist(grassCircle)

            if plot:
                plt.pause(0.1)
                plt.cla()
                axes.set_xlim(left=-90,right=90)
                axes.set_ylim(bottom=-90,top=90)

        if plot:
            plt.show()

        return len(self.preyPop), thingsPreyEat, numPreyBred, numPreyDied