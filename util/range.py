class Range(object):
    def __init__(self, min: float, max: float) -> None:
        self.min = min
        self.max = max

    def set_clamp(self, value: float) -> None:
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def clip(self, value: float) -> float:
        return min(max(value, self.min), self.max)

    def __repr__(self) -> str:
        return f"Range[{self.min:0.4f} {self.max:0.4f}]"
