from .climber import Climber


class Walker(Climber):
    def single_step(self):
        self.attempt += 1
        parent = self.attempts[-1][0]
        s = lambda: self.sample(parent)
        fitness, best = max([s() for _ in range(self.samples)])
        self.attempts.append((best, fitness))
        if fitness >= self.attempts[self.best][1]:
            self.best = self.attempt
        self.time += self.duration
