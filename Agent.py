import numpy as np
import Neural_Network

layers = [2, 1]
weightsRange = 5
biasRange = 5

class Agent:
    def __init__(self, xpos, ypos, velocity, angle, numSensors, sensorAngle):
        self.xpos = xpos
        self.ypos = ypos
        self.previousX = xpos
        self.previousY = ypos
        self.lastXPos = 0
        self.lastYPos = 0
        self.numSensors = numSensors
        self.velocity = velocity
        self.angle = angle
        self.sensorAngle = sensorAngle
        self.hunger = 140
        self.hasBred = False
        self.sensors = [0]*numSensors
        sensor1x = (np.cos(self.angle+self.sensorAngle+np.pi)*3)+xpos
        sensor1y = (np.sin(self.angle+self.sensorAngle+np.pi)*3)+ypos
        sensor2x = (np.cos(self.angle-self.sensorAngle+np.pi)*3)+xpos
        sensor2y = (np.sin(self.angle-self.sensorAngle+np.pi)*3)+ypos
        self.sensors[1] = [sensor1x, sensor1y]
        self.sensors[0] = [sensor2x, sensor2y]
        self.leftMotor = 1
        self.rightMotor = 1
        self.numEaten = 0
        self.outputHistory = []

    def walk(self, walkType, target = None, targetMate = None, neuralNet = None):
        self.lastXPos = self.xpos
        self.lastYPos = self.ypos
        if target == None:
            walkType = 'Random'

        match walkType:
            case 'Prey Neural Net':
                senses = []
                mateSenses = []
                #fix when no mate, make each inputValue seperate and then do 1 outputs statement
                for sensor in self.sensors:
                    distance = np.sqrt((sensor[0]-target[0])**2 + (sensor[1]-target[1])**2)
                    if targetMate is None:
                        mateDistance = -1
                    else:
                        mateDistance = np.sqrt((sensor[0]-targetMate.getXPos())**2 + (sensor[1]-targetMate.getYPos())**2)
                    senses.append(distance)
                    mateSenses.append(mateDistance)

                # leftFoodSesnse = (180-senses[0])/180
                # rightFoodSense = (180-senses[1])/180

                # if mateSenses[0] == -1 or mateSenses[1] == -1 or self.hasBred:
                #     leftMateSense = 0
                #     rightMateSense = 0
                # else:
                #     leftMateSense = (180-mateSenses[0])/180
                #     rightMateSense = (180-mateSenses[1])/180
                #outputs = neuralNet.step(dt = 0.1, inputs = [leftFoodSesnse, rightFoodSense, leftMateSense, rightMateSense])

                if self.hasBred:
                    outputs = neuralNet.step(dt = 0.1, inputs = [senses[0], senses[1], -1, -1])
                outputs = neuralNet.step(dt = 0.1, inputs = [senses[0], senses[1], mateSenses[0], mateSenses[1]])

                foodLeft = outputs[0]
                foodRight = outputs[1]
                mateLeft = outputs[2]
                mateRight = outputs[3]
                # breedWant = outputs[4]

                if foodLeft > mateLeft:
                    self.leftMotor = foodLeft
                else:
                    self.leftMotor = mateLeft
                if foodRight > mateRight:
                    self.rightMotor = foodRight
                else:
                    self.rightMotor = mateRight

                if self.leftMotor > self.rightMotor:
                    self.angle += np.pi/4
                else:
                    self.angle -= np.pi/4

                self.xpos += self.velocity * np.cos(self.angle)
                self.ypos += self.velocity * np.sin(self.angle)
                sensor1x = (np.cos(self.angle+self.sensorAngle+np.pi)*3)+self.xpos
                sensor1y = (np.sin(self.angle+self.sensorAngle+np.pi)*3)+self.ypos
                sensor2x = (np.cos(self.angle-self.sensorAngle+np.pi)*3)+self.xpos
                sensor2y = (np.sin(self.angle-self.sensorAngle+np.pi)*3)+self.ypos
                self.sensors[1] = [sensor1x, sensor1y]
                self.sensors[0] = [sensor2x, sensor2y]
                self.outputHistory.append(outputs)
            case 'Predator Neural Net':
                senses = []
                for sensor in self.sensors:
                    distance = np.sqrt((sensor[0]-target.getXPos())**2 + (sensor[1]-target.getYPos())**2)
                    senses.append(distance)

                if senses[0] > senses[1]:
                    outputs = neuralNet.step(dt = 0.1, inputs = [0.5, -0.5])
                else:
                    outputs = neuralNet.step(dt = 0.1, inputs = [-0.5, 0.5])
                #outputs = neuralNet.step(dt = 0.01, inputs = senses


                self.leftMotor = outputs[0]
                self.rightMotor = outputs[1]

                if self.leftMotor > self.rightMotor:
                    self.angle += np.pi/4
                else:
                    self.angle -= np.pi/4

                self.xpos += self.velocity * np.cos(self.angle)
                self.ypos += self.velocity * np.sin(self.angle)
                sensor1x = (np.cos(self.angle+self.sensorAngle+np.pi)*3)+self.xpos
                sensor1y = (np.sin(self.angle+self.sensorAngle+np.pi)*3)+self.ypos
                sensor2x = (np.cos(self.angle-self.sensorAngle+np.pi)*3)+self.xpos
                sensor2y = (np.sin(self.angle-self.sensorAngle+np.pi)*3)+self.ypos
                self.sensors[1] = [sensor1x, sensor1y]
                self.sensors[0] = [sensor2x, sensor2y]
            case 'Random':
                self.angle = np.random.random() * np.pi * 2
                self.xpos += self.velocity * np.cos(self.angle)
                self.ypos += self.velocity * np.sin(self.angle)
                for sensor in self.sensors:
                    sensor[0] += self.velocity * np.cos(self.angle)
                    sensor[1] += self.velocity * np.sin(self.angle)

        if self.xpos > 90 or self.xpos < -90:
            self.angle += np.pi
        if self.ypos > 90 or self.ypos < -90:
            self.angle += np.pi
        self.hunger -= 1
    
    def getXPos(self):
        return self.xpos
    
    def getYPos(self):
        return self.ypos
    
    def getPreviousXPos(self):
        return self.previousX
    
    def getPreviousYPos(self):
        return self.previousY
    
    def getHunger(self):
        return self.hunger
    
    def eat(self):
        self.hunger += 25
        self.numEaten += 1

    def getHasBred(self):
        return self.hasBred

    def breed(self):
        self.hasBred = True

    def getSensors(self):
        return self.sensors
    
    def getNumEaten(self):
        return self.numEaten
    
    def getOutputHistory(self):
        return self.outputHistory