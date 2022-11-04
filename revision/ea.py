import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
from learningFunction import learn
import concurrent.futures
import time
import numpy as np

data = np.zeros(20)
durations = np.random.randint(0, 20, 20)


def getIndex(idx, genome, params=None):
    print(idx, end=" ", flush=False)
    end_fit = learn(genome, **params)
    return (idx, end_fit)


class Microbial:
    def __init__(
        self,
        fitnessFunction,
        popsize,
        genesize,
        recombProb,
        mutatProb,
        demeSize,
        generations,
        generator_type,
        neuron_configuration,
        size,
        num_processes=1,
        params=None,
        learning=True,
    ):
        self.fitnessFunction = fitnessFunction
        self.popsize = popsize
        self.genesize = genesize
        self.recombProb = recombProb
        self.mutatProb = mutatProb
        self.demeSize = int(demeSize / 2)
        self.generations = generations
        self.tournaments = generations * popsize
        self.pop = np.random.rand(popsize, genesize) * 2 - 1
        self.learned_fitness = np.zeros(popsize)
        self.fitness = np.zeros(popsize)
        self.avgHistory = np.zeros(generations)
        self.size = size
        self.bestHistory = np.zeros(generations)
        self.generator_type = generator_type
        self.neuron_configuration = neuron_configuration
        self.gen = 0
        self.num_processes = num_processes
        self.params = params

    def getFitnessIndex(self, idx, genome):
        self.fitness[idx] = self.fitnessFunction(
            self.pop[idx],
            N=params["size"],
            generate_type=self.params["generator_type"],
            configuration=self.params["neuron_configuration"],
        )

    def showFitness(self, label="", c="k", save=False):
        plt.plot(
            self.bestHistory,
            label=label + " bestHist " + self.generator_type,
            color=c,
            ls="dashed",
        )
        plt.plot(
            self.avgHistory,
            label=label + " averageHist " + self.generator_type,
            color=c,
            ls="solid",
        )
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.legend()
        if save:
            filename = f"{self.generator_type}-s{self.size}-c{'_'.join(str(num) for num in self.neuron_configuration)}"
            plt.savefig(Path(f"./data/microbial/{filename}"))

    def fitStats(self):
        bestind = self.pop[np.argmax(self.learned_fitness)]
        bestfit = np.max(self.learned_fitness)
        avgfit = np.mean(self.learned_fitness)
        self.avgHistory[self.gen] = avgfit
        self.bestHistory[self.gen] = bestfit
        return avgfit, bestfit, bestind

    def save(self, filename=None):
        af, bf, bi = self.fitStats()
        if not filename:
            filename = f"{self.generator_type}-s{self.size}-c{'_'.join(str(num) for num in self.neuron_configuration)}"
        np.savez(
            Path(f"./data/microbial/genomes/{filename}"),
            avghist=self.avgHistory,
            besthist=self.bestHistory,
            bestind=bi,
        )

    def run(self):
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_processes
        ) as executor:
            # Calculate all fitness once
            print("init agent:")
            for i in range(self.popsize):
                self.fitness[i] = self.fitnessFunction(
                    self.pop[i],
                    N=self.size,
                    generator_type=self.generator_type,
                    configuration=self.neuron_configuration,
                )
                results.append(
                    executor.submit(
                        getIndex,
                        i,
                        self.pop[i],
                        params=self.params,
                    )
                )
                if len(results) == self.num_processes:
                    for future in concurrent.futures.as_completed(results):
                        idx, end_fit = future.result()
                        self.learned_fitness[idx] = end_fit
                        results = []
                    print()
            for i, future in enumerate(concurrent.futures.as_completed(results)):
                idx, end_fit = future.result()
                self.learned_fitness[idx] = end_fit
                results = []
            # Evolutionary loop
            print("\ngeneration:")
            for g in range(self.generations):
                print(f"Generation: {g}")
                for i in range(self.popsize):
                    print(
                        f"{i}: {np.round(self.fitness[i],3)}, {np.round(self.learned_fitness[i], 3)}"
                    )
                self.gen = g
                # Report statistics every generation
                self.fitStats()
                # print("Evaluations:")
                # print(f"mean:{np.mean(self.learned_fitness)}|max:{max(self.fitness)}")
                for i in range(self.popsize):
                    # Step 1: Pick 2 individuals
                    a = np.random.randint(0, self.popsize - 1)
                    b = (
                        np.random.randint(a - self.demeSize, a + self.demeSize - 1)
                        % self.popsize
                    )  ### Restrict to demes
                    while a == b:  # Make sure they are two different individuals
                        b = (
                            np.random.randint(a - self.demeSize, a + self.demeSize - 1)
                            % self.popsize
                        )  ### Restrict to demes
                    # Step 2: Compare their fitness
                    if self.learned_fitness[a] > self.learned_fitness[b]:
                        winner = a
                        loser = b
                    else:
                        winner = b
                        loser = a
                    # Step 3: Transfect loser with winner --- Could be made more efficient using Numpy
                    for j in range(self.genesize):
                        if np.random.random() < self.recombProb:
                            self.pop[loser][j] = self.pop[winner][j]
                    # Step 4: Mutate loser and make sure new organism stays within bounds
                    self.pop[loser] += np.random.normal(
                        0.0, self.mutatProb, size=self.genesize
                    )
                    self.pop[loser] = np.clip(self.pop[loser], -1, 1)
                    # Save fitness
                    #
                    results.append(
                        executor.submit(
                            getIndex,
                            loser,
                            self.pop[loser],
                            params=self.params,
                        )
                    )
                    if (len(results) == self.num_processes) or (i + 1 == self.popsize):
                        for future in concurrent.futures.as_completed(results):
                            idx, end_fit = future.result()
                            self.learned_fitness[idx] = end_fit
                            results = []
                    self.fitness[loser] = self.fitnessFunction(
                        self.pop[loser],
                        N=self.size,
                        generator_type=self.generator_type,
                        configuration=self.neuron_configuration,
                    )


class Microbial2:
    def __init__(
        self,
        fitnessFunction,
        popsize,
        genesize,
        recombProb,
        mutatProb,
        demeSize,
        generations,
        generator_type,
        neuron_configuration,
        size,
    ):
        self.fitnessFunction = fitnessFunction
        self.popsize = popsize
        self.genesize = genesize
        self.recombProb = recombProb
        self.mutatProb = mutatProb
        self.demeSize = int(demeSize / 2)
        self.generations = generations
        self.tournaments = generations * popsize
        self.pop = np.random.rand(popsize, genesize) * 2 - 1
        self.fitness = np.zeros(popsize)
        self.avgHistory = np.zeros(generations)
        self.size = size
        self.bestHistory = np.zeros(generations)
        self.generator_type = generator_type
        self.neuron_configuration = neuron_configuration
        self.gen = 0

    def showFitness(self, label="", c="k", save=False):
        plt.plot(
            self.bestHistory,
            label=label + " bestHist " + self.generator_type,
            color=c,
            ls="dashed",
        )
        plt.plot(
            self.avgHistory,
            label=label + " averageHist " + self.generator_type,
            color=c,
            ls="solid",
        )
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.legend()
        if save:
            filename = f"{self.generator_type}-s{self.size}-c{'_'.join(str(num) for num in self.neuron_configuration)}"
            plt.savefig(Path(f"./data/microbial/{filename}"))

    def fitStats(self):
        bestind = self.pop[np.argmax(self.fitness)]
        bestfit = np.max(self.fitness)
        avgfit = np.mean(self.fitness)
        self.avgHistory[self.gen] = avgfit
        self.bestHistory[self.gen] = bestfit
        return avgfit, bestfit, bestind

    def save(self, filename=None):
        af, bf, bi = self.fitStats()
        if not filename:
            filename = f"{self.generator_type}-s{self.size}-c{'_'.join(str(num) for num in self.neuron_configuration)}"
        np.savez(
            Path(f"./data/microbial/genomes/{filename}"),
            avghist=self.avgHistory,
            besthist=self.bestHistory,
            bestind=bi,
        )

    def run(self):
        # Calculate all fitness once
        print("init agent:")
        for i in range(self.popsize):
            print(f"{i}", end=" ", flush=False)
            self.fitness[i] = self.fitnessFunction(
                self.pop[i],
                N=self.size,
                generator_type=self.generator_type,
                configuration=self.neuron_configuration,
            )
        # Evolutionary loop
        print("\ngeneration:")
        for g in range(self.generations):
            print(f"{g}", end=" ", flush=False)
            self.gen = g

            # Report statistics every generation
            self.fitStats()
            # print("Evaluations:")
            # print(f"mean:{np.mean(self.fitness)}|max:{max(self.fitness)}")
            for i in range(self.popsize):
                # Step 1: Pick 2 individuals
                a = np.random.randint(0, self.popsize - 1)
                b = (
                    np.random.randint(a - self.demeSize, a + self.demeSize - 1)
                    % self.popsize
                )  ### Restrict to demes
                while a == b:  # Make sure they are two different individuals
                    b = (
                        np.random.randint(a - self.demeSize, a + self.demeSize - 1)
                        % self.popsize
                    )  ### Restrict to demes
                # Step 2: Compare their fitness
                if self.fitness[a] > self.fitness[b]:
                    winner = a
                    loser = b
                else:
                    winner = b
                    loser = a
                # Step 3: Transfect loser with winner --- Could be made more efficient using Numpy
                for l in range(self.genesize):
                    if np.random.random() < self.recombProb:
                        self.pop[loser][l] = self.pop[winner][l]
                # Step 4: Mutate loser and make sure new organism stays within bounds
                self.pop[loser] += np.random.normal(
                    0.0, self.mutatProb, size=self.genesize
                )
                self.pop[loser] = np.clip(self.pop[loser], -1, 1)
                # Save fitness
                #
                self.fitness[loser] = self.fitnessFunction(
                    self.pop[loser],
                    N=self.size,
                    generator_type=self.generator_type,
                    configuration=self.neuron_configuration,
                )


class Microbial2:
    def __init__(
        self,
        fitnessFunction,
        popsize,
        genesize,
        recombProb,
        mutatProb,
        demeSize,
        generations,
        generator_type,
        neuron_configuration,
        size,
    ):
        self.fitnessFunction = fitnessFunction
        self.popsize = popsize
        self.genesize = genesize
        self.recombProb = recombProb
        self.mutatProb = mutatProb
        self.demeSize = int(demeSize / 2)
        self.generations = generations
        self.tournaments = generations * popsize
        self.pop = np.random.rand(popsize, genesize) * 2 - 1
        self.fitness = np.zeros(popsize)
        self.avgHistory = np.zeros(generations)
        self.size = size
        self.bestHistory = np.zeros(generations)
        self.generator_type = generator_type
        self.neuron_configuration = neuron_configuration
        self.gen = 0

    def showFitness(self, label="", c="k", save=False):
        plt.plot(
            self.bestHistory,
            label=label + " bestHist " + self.generator_type,
            color=c,
            ls="dashed",
        )
        plt.plot(
            self.avgHistory,
            label=label + " averageHist " + self.generator_type,
            color=c,
            ls="solid",
        )
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.legend()
        if save:
            filename = f"{self.generator_type}-s{self.size}-c{'_'.join(str(num) for num in self.neuron_configuration)}"
            plt.savefig(Path(f"./data/microbial/{filename}"))

    def fitStats(self):
        bestind = self.pop[np.argmax(self.fitness)]
        bestfit = np.max(self.fitness)
        avgfit = np.mean(self.fitness)
        self.avgHistory[self.gen] = avgfit
        self.bestHistory[self.gen] = bestfit
        return avgfit, bestfit, bestind

    def save(self, filename=None):
        af, bf, bi = self.fitStats()
        if not filename:
            filename = f"{self.generator_type}-s{self.size}-c{'_'.join(str(num) for num in self.neuron_configuration)}"
        np.savez(
            Path(f"./data/microbial/genomes/{filename}"),
            avghist=self.avgHistory,
            besthist=self.bestHistory,
            bestind=bi,
        )

    def run(self):
        # Calculate all fitness once
        print("init agent:")
        for i in range(self.popsize):
            print(f"{i}", end=" ", flush=False)
            self.fitness[i] = self.fitnessFunction(
                self.pop[i],
                N=self.size,
                generator_type=self.generator_type,
                configuration=self.neuron_configuration,
            )
        # Evolutionary loop
        print("\ngeneration:")
        for g in range(self.generations):
            print(f"{g}", end=" ", flush=False)
            self.gen = g

            # Report statistics every generation
            self.fitStats()
            # print("Evaluations:")
            # print(f"mean:{np.mean(self.fitness)}|max:{max(self.fitness)}")
            for i in range(self.popsize):
                # Step 1: Pick 2 individuals
                a = np.random.randint(0, self.popsize - 1)
                b = (
                    np.random.randint(a - self.demeSize, a + self.demeSize - 1)
                    % self.popsize
                )  ### Restrict to demes
                while a == b:  # Make sure they are two different individuals
                    b = (
                        np.random.randint(a - self.demeSize, a + self.demeSize - 1)
                        % self.popsize
                    )  ### Restrict to demes
                # Step 2: Compare their fitness
                if self.fitness[a] > self.fitness[b]:
                    winner = a
                    loser = b
                else:
                    winner = b
                    loser = a
                # Step 3: Transfect loser with winner --- Could be made more efficient using Numpy
                for l in range(self.genesize):
                    if np.random.random() < self.recombProb:
                        self.pop[loser][l] = self.pop[winner][l]
                # Step 4: Mutate loser and make sure new organism stays within bounds
                self.pop[loser] += np.random.normal(
                    0.0, self.mutatProb, size=self.genesize
                )
                self.pop[loser] = np.clip(self.pop[loser], -1, 1)
                # Save fitness
                #
                self.fitness[loser] = self.fitnessFunction(
                    self.pop[loser],
                    N=self.size,
                    generator_type=self.generator_type,
                    configuration=self.neuron_configuration,
                )
