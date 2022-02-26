import random
import numpy as np
import time
import matplotlib.pyplot as plt

# Note: all random numbers are generated using np.random, so to make this reproducible
# be sure to set np.random.seed() BEFORE constructing an instance of this class

class MicrobialGA():
    def __init__(self, fitnessFunction, popsize, genesize, recombProb, mutatProb ):
        self.fitnessFunction = fitnessFunction
        self.popsize = popsize
        self.genesize = genesize
        self.recombProb = recombProb
        self.mutatProb = mutatProb
        self.pop = np.random.rand(popsize,genesize)*2 - 1
        self.fitness = np.zeros(popsize)
        self.changed = np.ones(popsize)

    def showFitness(self, save=False, path=None):
        plt.figure()
        if len(self.bestHistory) == 1:
            plt.scatter(0, self.bestHistory)
            plt.scatter(0, self.avgHistory)
        else:
            plt.plot(self.bestHistory)
            plt.plot(self.avgHistory)
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best and average fitness")
        if save:
            plt.savefig(path)
        else:
            plt.show()
    
    # Set show_progress to be False to not show progress of run
    def fitStats(self, show_progress = True):
        bestind = self.pop[np.argmax(self.fitness)]
        bestfit = np.max(self.fitness)
        avgfit = np.mean(self.fitness)
        
        if show_progress:
            print( f"{self.gen}  avg:{avgfit}  best:{bestfit}  {self.predict_time_remaining()}" )

        self.avgHistory[self.gen]=avgfit
        self.bestHistory[self.gen]=bestfit
        return avgfit, bestfit, bestind

    # NOT a true generational loop, instead randomly selected competitions (tournament)
    def run_tournaments(self, tournaments, show_progress=True):
        if self.popsize ==1:
            print("Cannot run this algorithm with popsize=1, exiting...")
            quit()
        self.start = time.time()
        self.gen = 0
        self.generations = int(tournaments / self.popsize)  + 1  # +1 for generation 0
        self.avgHistory = np.zeros(self.generations )    
        self.bestHistory = np.zeros(self.generations)

        #evaluate fitness BEFORE evolution begins
        for i in range(self.popsize):
            self.fitness[i] = self.fitnessFunction(self.pop[i])

        # A single tournament really means no genetic operations, 
        # but still evaluations of the fitness of individuals
        if tournaments == 1:
            af, bf, bi = self.fitStats(show_progress=show_progress)
            return

        # Evolutionary loop
        for i in range(tournaments):
            # Report/Record statistics every generation
            if (i%self.popsize==0):
                af, bf, bi = self.fitStats(show_progress=show_progress)
                self.gen += 1

            # Step 1: Pick 2 individuals
            a = np.random.randint(0,self.popsize)  #Note np second parameter here is EXCLUSIVE
            b = np.random.randint(0,self.popsize)  #Note np second parameter here is EXCLUSIVE
            while (a==b):   # Make sure they are two different individuals
                b = np.random.randint(0,self.popsize)  #Note np second parameter here is EXCLUSIVE


            # Step 2: Compare their fitness
            if (self.fitnessFunction(self.pop[a]) > self.fitnessFunction(self.pop[b])):
                winner = a
                loser = b
            else:
                winner = b
                loser = a

            # Step 3: Transfect loser with winner
            for l in range(self.genesize):
                if (np.random.random() < self.recombProb):
                    self.pop[loser][l] = self.pop[winner][l]

            # Step 4: Mutate loser and Make sure new organism stays within bounds
            for l in range(self.genesize):
                self.pop[loser][l] += np.random.normal(0.0,self.mutatProb)
                if self.pop[loser][l] > 1.0:
                    self.pop[loser][l] = 1.0
                if self.pop[loser][l] < -1.0:
                    self.pop[loser][l] = -1.0
            
            # Save fitness
            self.fitness[loser] = self.fitnessFunction(self.pop[loser])
            self.changed[loser] = 0
    
    # Run true generational EA
    # Optional terminal best fitness value to stop loop early
    def run_evolution(self, generations, term_fitness=1.0, show_progress=True):
        self.start = time.time()
        self.gen = 0
        self.generations = generations + 1  # +1 for generation 0
        self.avgHistory = np.zeros(self.generations)
        self.bestHistory = np.zeros(self.generations)
        # Calculate all fitness once
        for i in range(self.popsize):
            self.fitness[i] = self.fitnessFunction(self.pop[i])
        
        # Evolutionary loop
        for g in range( generations):
            self.gen = g
            # Report statistics every generation
            avgfit, bestfit, bestind = self.fitStats(show_progress=show_progress)
            if bestfit > term_fitness:
                return self.gen

            self.changed = np.ones(self.popsize)
            for i in range(self.popsize):
                # Step 1: Pick 2 individuals
                a = np.random.randint(0,self.popsize-1)
                b = np.random.randint(0,self.popsize-1)
                while (a==b):   # Make sure they are two different individuals
                    b = np.random.randint(0,self.popsize-1)
                # Step 2: Compare their fitness
                if (self.fitness[a] > self.fitness[b]):
                    winner = a
                    loser = b
                else:
                    winner = b
                    loser = a
                # Step 3: Transfect loser with winner
                for l in range(self.genesize):
                    if (np.random.random() < self.recombProb):
                        self.pop[loser][l] = self.pop[winner][l]
                # Step 4: Mutate loser and Make sure new organism stays within bounds
                for l in range(self.genesize):
                    self.pop[loser][l] += np.random.normal(0.0,self.mutatProb)
                    if self.pop[loser][l] > 1.0:
                        self.pop[loser][l] = 1.0
                    if self.pop[loser][l] < -1.0:
                        self.pop[loser][l] = -1.0
                # Save fitness
                self.fitness[loser] = self.fitnessFunction(self.pop[loser])
                self.changed[loser] = 0
        
        #return number of generations taken (only less than generations, if term_fitness reached)
        return self.gen

    def predict_time_remaining(self ):
        timeElapsed = time.time() - self.start
        averageTimePerRep = timeElapsed / ( self.gen + 1)
        totalTime = averageTimePerRep * self.generations
        timeLeft = int( self.start + totalTime - time.time() )
        hours = int(timeLeft / 3600)
        minutes = int( (timeLeft - 3600 * hours) / 60)
        seconds = int( (timeLeft - 3600 * hours) - 60 * minutes)
        return f"Estimated Time Remaining (hh:mm:ss)   {hours}:{minutes}:{seconds}"
