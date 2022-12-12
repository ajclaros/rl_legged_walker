import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import concurrent.futures
import time
import numpy as np

data = np.zeros(20)
durations = np.random.randint(0, 20, 20)


def getIndex(idx, genome, params=None, num_trials=1):
    fit = []
    perf = []
    for i in range(num_trials):
        end_fit, end_perf = learn(
            genome,
            **params,
        )
        fit.append(end_fit)
        perf.append(end_perf)
        # if end_fit > max_fit:
        #     max_fit = end_fit
        #     max_perf = end_perf

    return (idx, np.median(fit), np.median(perf))


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
        verbose=None,
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
        self.verbose = verbose
        self.bestHistory = np.zeros(generations)
        self.generator_type = generator_type
        self.neuron_configuration = neuron_configuration
        self.neuron_conf_str = list(map(str, self.neuron_configuration))
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

    def run(self, savenp):
        # Calculate all fitness once
        if self.verbose:
            print("init agent:")
        for i in range(self.popsize):
            if self.verbose:
                print(f"{i}", end=" ", flush=False)
            self.fitness[i] = self.fitnessFunction(
                self.pop[i],
                N=self.size,
                generator_type=self.generator_type,
                configuration=self.neuron_configuration,
            )
        # Evolutionary loop
        if self.verbose:
            print("\ngeneration:")
        for g in range(self.generations):
            max_fit = np.argmax(self.fitness)
            if self.verbose:
                print(f"Max fit: {self.fitness[max_fit]}")

            fit_string = str(int(self.fitness[max_fit] * 100000))
            fit_string = fit_string.zfill(5)
            np.save(
                f"./evolved/{self.generator_type}/{self.size}/{''.join(self.neuron_conf_str)}/fit-{fit_string}",
                self.pop[max_fit],
            )

            if self.verbose:
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


class GaEliteLearn:
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
        num_trials=1,
    ):
        self.num_trials = num_trials
        self.fitnessFunction = fitnessFunction
        self.popsize = popsize
        self.genesize = genesize
        self.recombProb = recombProb
        self.mutatProb = mutatProb
        self.mutateVar = mutatProb
        self.demeSize = int(demeSize / 2)
        self.generations = generations
        self.tournaments = generations * popsize
        self.pop = np.random.rand(popsize, genesize) * 2 - 1
        self.fitness = np.zeros(popsize)
        self.learned_fitness = np.zeros(popsize)
        self.performance = np.zeros(popsize)
        self.avgHistory = np.zeros(generations)
        self.avgPerf = np.zeros(generations)
        self.bestPerf = np.zeros(generations)
        self.bestLearnedFitness = np.zeros(generations)
        self.average_learned_fit = np.zeros(generations)
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
            generator_type=self.params["generator_type"],
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
        plt.plot(
            self.avgPerf,
            label=label + " averagePerf" + self.generator_type,
            color=c,
            ls="dashdot",
        )

        plt.plot(
            self.bestPerf,
            label=label + " bestPerf" + self.generator_type,
            color=c,
            ls="dotted",
        )

        plt.plot(
            self.bestLearnedFitness,
            label=label + "fitness after learning" + self.generator_type,
            color=c,
            marker="o",
        )
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.legend()
        if save:
            filename = f"{self.generator_type}-s{self.size}-c{'_'.join(str(num) for num in self.neuron_configuration)}"
            plt.savefig(Path(f"./data/microbial/{filename}"))

    def fitStats(self):
        bestind = np.argmax(self.learned_fitness)
        best_genome = self.pop[np.argmax(self.learned_fitness)]
        bestfit = np.max(self.fitness)
        learnedFit = self.learned_fitness[bestind]
        average_learned_fit = self.learned_fitness.mean()
        avgfit = np.mean(self.fitness)
        best_perf = self.performance[bestind]
        avg_perf = np.mean(self.performance)
        print(f"Generation:{self.gen}")
        print("Best")
        print("------------------")
        print(f"idx:{bestind}")
        print("fitness:")
        print(self.fitness[bestind])
        print("learned fitness :")
        print(self.learned_fitness[bestind])
        print("performance:")
        print(self.performance[bestind])
        print("genome")
        print(best_genome)
        self.avgHistory[self.gen] = avgfit
        self.bestHistory[self.gen] = bestfit
        self.bestPerf[self.gen] = best_perf
        self.avgPerf[self.gen] = avg_perf
        self.bestLearnedFitness[self.gen] = learnedFit
        self.average_learned_fit[self.gen] = average_learned_fit
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
                np.random.seed(np.random.randint(10000))
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
                        num_trials=self.num_trials,
                    )
                )
                if len(results) == self.num_processes:
                    for future in concurrent.futures.as_completed(results):
                        idx, end_fit, end_perf = future.result()
                        self.learned_fitness[idx] = end_fit
                        self.performance[idx] = end_perf
                        results = []
                    print()
            for i, future in enumerate(concurrent.futures.as_completed(results)):
                idx, end_fit, end_perf = future.result()
                self.learned_fitness[idx] = end_fit
                self.performance[idx] = end_perf
                results = []
            # Evolutionary loop
            print("\ngeneration:")
            for g in range(self.generations):
                print(f"Generation: {g}")
                print("Start fitness, performance, end fitness")
                for i in range(self.popsize):
                    print(
                        f"{i}: {np.round(self.fitness[i],5)}, {np.round(self.performance[i], 5)}, {np.round(self.learned_fitness[i], 5)}"
                    )
                self.gen = g
                # Report statistics every generation
                self.fitStats()
                max_fit = np.argmax(self.learned_fitness)
                # print("Evaluations:")
                # print(f"mean:{np.mean(self.learned_fitness)}|max:{max(self.fitness)}")
                for i in range(self.popsize):
                    np.random.seed(np.random.randint(10000))
                    # Step 1: Pick 2 individuals
                    if i != max_fit:
                        for j in range(self.genesize):
                            if np.random.random() < self.recombProb:
                                self.pop[i][j] = self.pop[max_fit][j]
                        # Step 4: Mutate loser and make sure new organism stays within bounds
                        magnitude = np.random.normal(0, self.mutateVar)
                        temp = np.random.rand(self.genesize)
                        temp = temp / np.linalg.norm(temp)
                        self.pop[i] = np.clip(self.pop[i] + magnitude * temp, -1, 1)

                        # self.pop[i] += np.random.normal(
                        #     0.0, self.mutatProb, size=self.genesize
                        # )
                        # self.pop[i] = np.clip(self.pop[i], -1, 1)
                    # Save fitness
                    results.append(
                        executor.submit(
                            getIndex,
                            i,
                            self.pop[i],
                            params=self.params,
                            num_trials=self.num_trials,
                        )
                    )
                    if (len(results) == self.num_processes) or (i + 1 == self.popsize):
                        for future in concurrent.futures.as_completed(results):
                            idx, end_fit, end_perf = future.result()
                            self.learned_fitness[idx] = end_fit
                            self.performance[idx] = end_perf
                            results = []
                    self.fitness[i] = self.fitnessFunction(
                        self.pop[i],
                        N=self.size,
                        generator_type=self.generator_type,
                        configuration=self.neuron_configuration,
                    )


class GaElite:
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
        verbose=None,
    ):
        self.fitnessFunction = fitnessFunction
        self.popsize = popsize
        self.genesize = genesize
        self.recombProb = recombProb
        self.mutatProb = mutatProb
        self.mutateVar = mutatProb
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
        self.neuron_conf_str = list(map(str, self.neuron_configuration))
        self.gen = 0
        self.verbose = verbose

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
            filename = f"{self.generator_type}-s{self.size}-c{'_'.join(str(num) for num in self.neuron_conf_str)}"
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

    def run(self, savenp):
        # Calculate all fitness once
        for i in range(self.popsize):
            print(f"{i}", end=" ", flush=False)
            self.fitness[i] = self.fitnessFunction(
                self.pop[i],
                N=self.size,
                generator_type=self.generator_type,
                configuration=self.neuron_configuration,
            )
        # Evolutionary loop
        if self.verbose:
            print("\ngeneration:")
        for g in range(self.generations):
            if self.verbose:
                print(f"Generation: {g}")
            self.gen = g

            # Report statistics every generation
            self.fitStats()
            # print("Evaluations:")
            # print(f"mean:{np.mean(self.fitness)}|max:{max(self.fitness)}")

            max_fit = np.argmax(self.fitness)
            if self.verbose:
                print(f"Max fit: {self.fitness[max_fit]}")
            fit_string = str(int(self.fitness[max_fit] * 100000))
            fit_string = fit_string.zfill(5)

            pathname = f"./evolved/{self.generator_type}/{self.size}/{''.join(self.neuron_conf_str)}/fit-{fit_string}"

            print(f"Max fitness:{self.fitness[max_fit]}")
            print(f"Saved to: {pathname}")
            np.save(
                pathname,
                self.pop[max_fit],
            )
            for i in range(self.popsize):
                if i == max_fit:
                    continue
                for j in range(self.genesize):
                    if np.random.random() < self.recombProb:
                        self.pop[i][j] = self.pop[max_fit][j]
                # Step 4: Mutate loser and make sure new organism stays within bounds

                magnitude = np.random.normal(0, self.mutateVar)
                temp = np.random.rand(self.genesize)
                np.random.rand
                temp = temp / np.linalg.norm(temp)
                self.pop[i] = np.clip(self.pop[i] + magnitude * temp, -1, 1)
                # self.pop[i] += np.random.normal(0.0, self.mutatProb, size=self.genesize)
                # self.pop[i] = np.clip(self.pop[i], -1, 1)
                # Save fitness
                self.fitness[i] = self.fitnessFunction(
                    self.pop[i],
                    N=self.size,
                    generator_type=self.generator_type,
                    configuration=self.neuron_configuration,
                )
