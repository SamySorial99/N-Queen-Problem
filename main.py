import numpy as np
import sys
import random

np.set_printoptions(threshold=sys.maxsize)

# to show all the results of really long arrays without dots (play with it)
numOfBoard = input("Enter dimension of the board (N):")
populationNumber = input("Enter number of population for the genetic algorithm:")
x = input("Enter weight to choose parents from, weight of parents will be smaller than or equal entered value: ")
population = np.empty([int(populationNumber), int(numOfBoard)])
finalResult = np.array([])

# populating the array with random numbers
def randomGen(population, numOfBoard):
    for i in range(len(population)):
        x = np.random.choice(range(int(numOfBoard)), int(numOfBoard), replace=False)
        population[i, :] = x


randomGen(population, numOfBoard)
population = population + 1


# heuristic function of the n-Queen problem


def heuristic(population, numOfBoard):
    weightFunctionElements = np.copy(population)  # by value not refrence
    for i in range(len(population)):  # for rows
        for k in range(0, int(numOfBoard)):  # for coulombs
            flag = 0
            y = np.copy(population[i][k])
            for j in range(1, int(numOfBoard) - k):  # loop check the value of next coulombs diagonally
                if y == 1:  # if it's in first square we can't check the next coulomb diagonally up
                    if y == population[i][k + j] - j:
                        flag = flag + 1
                elif y == int(numOfBoard):  # if it's in the last cell we can't check the next coulomb diagonally down
                    if y == population[i][k + j] + j:
                        flag = flag + 1
                elif y == population[i][k + j] - j or y == population[i][k + j] + j:
                    flag = flag + 1
            weightFunctionElements[i][k] = flag  # fill the array with heuristic value for each element
    return weightFunctionElements


def parentSearch(weightFunction, x, population):
    parentsindex = np.array([])
    for i in range(0, populationNumber):
        if int(weightFunction[i]) <= int(x):
            parentsindex = np.concatenate([parentsindex, [i]])
    parentsResult = np.empty([parentsindex.shape[0],population.shape[1]])
    shit = 0
    for j in parentsindex:
        parentsResult[int(shit)] = population[int(j)]
        shit = shit +1
    return parentsResult


def resultSearch(weightFunction,population):
    resultindex = np.array([])
    for i in range(0,int(weightFunction.size)):
        if int(weightFunction[i]) == 0:
            resultindex = np.concatenate([resultindex, [i]])
    finalresult = np.empty([resultindex.shape[0],population.shape[1]])
    shit = 0
    for j in resultindex:
        finalresult[int(shit)] = population[int(j)]
        shit = shit + 1
    return finalresult


def crossOver(parentsResult):
    parent1index = random.randint(0, parentsResult.shape[0])
    parent2index = random.randint(0, parentsResult.shape[0])
    while parent1index == parent2index:
        parent2index = random.randint(0, parentsResult.shape[0])
    range1 = random.randint(0, parentsResult.shape[1])
    range2 = random.randint(range1, parentsResult.shape[1])
    while range1 == range2:
        range2 = random.randint(range1 + 1, parentsResult.shape[1])
    parent1 = np.copy(parentsResult[parent1index])
    parent2 = np.copy(parentsResult[parent2index])
    segment = np.copy(parent1[range1:range2])
    for i in segment:
        parent2 = np.delete(parent2, np.where(parent2 == i))
    parent2 = np.insert(parent2, range1, segment)
    return parent2


def mutation(parentsResult):
    parent1index = random.randint(0, parentsResult.shape[0])
    range1 = random.sample(range(0,parentsResult.shape[1]),2)
    #print("RANGE IN MUTATION",range1,"ON PARENT NUMBER",parent1index-1)
    parent = np.copy(parentsResult[parent1index-1])
    temp = parent[range1[0]]
    parent[range1[0]] = parent[range1[1]]
    parent[range1[1]] = temp
    return parent


itterations = 0

while finalResult.size == 0:
    #population = np.unique(population, axis=0)
    populationNumber = np.shape(population)[0]

    weightFunctionElements = heuristic(population, numOfBoard)
    weightFunction = np.sum(weightFunctionElements, axis=1)

    parentsresult = parentSearch(weightFunction, x, population)
    finalResult = resultSearch(weightFunction, population)
    #print("population :\n",population)
    #print("the weight of every chromosome\n",weightFunction)
    #print("The chosen parents\n",parentsresult)
    mutated = mutation(parentsresult)
    itterations = itterations + 1
    highestweightindex = np.where(weightFunction == np.amax(weightFunction))
    print(highestweightindex[0][0])
    population[highestweightindex[0][0]] = np.copy(mutated)
    print("INSIDE LOOP",itterations)
print("\aFinal result\n",finalResult)
print("Number of iteration",itterations)