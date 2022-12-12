import numpy as np
import numpy.typing as npt


class simulator2:
    def __init__(
        self,
        duration: int = 2000,
        learn_rate: float = 0.1,
        conv_rate: float = 0.1,
        delay: int = 0.1,
        center_mat: npt.NDArray = np.zeros((2, 2)),
        p_min: npt.NDArray = np.zeros((2, 2)),
        p_max: npt.NDArray = np.zeros((2, 2)),
        period_min: npt.NDArray = np.zeros((2, 2)),
        period_max: npt.NDArray = np.zeros((2, 2)),
        init_flux: npt.NDArray = np.zeros((2, 2)),
        max_flux: npt.NDArray = np.zeros((2, 2)),
    ) -> None:
        self.duration = duration
        self.learn_rate = learn_rate
        self.conv_rate = conv_rate

        # variables for weight centers and weight fluctuations
        self.center_mat = center_mat
        self.flux_displacements = np.zeros(center_mat.shape)
        self.extended_mat = self.center_mat.copy()
        # Enables specific tuning of period and flux ranges for a heterogenous system
        # p_min, p_max are bounds on a parameter's
        self.size = self.center_mat.shape[1]
        self.flux_per_node = self.center_mat.shape[0]
        self.p_min = p_min
        self.p_max = p_max
        self.period_min = period_min
        self.period_max = period_max
        self.max_flux = max_flux
        self.init_flux = init_flux
        self.flux_mat = self.init_flux.copy()
        self.moments = np.zeros(center_mat.shape)
        self.periods = np.zeros(center_mat.shape)
        self.flux_sign = np.random.binomial(1, 0.5, size=(self.size, self.size)) * 2 - 1

    def iter_moment(self, dt):
        self.moments += dt
        new_period = np.where(self.moments > abs(self.periods))
        self.moments[new_period] = 0
        self.periods[new_period] = np.random.uniform(
            low=self.period_min[new_period],
            high=self.period_max[new_period],
            size=new_period[0].size,
        )

    def update_weights_with_reward(self, reward, tolerance=0.0, learning=True):
        self.reward = reward
        # update fluctuation size and weight centers
        if abs(reward) >= tolerance and learning:
            # print(self.conv_rate * reward)
            self.flux_mat -= self.conv_rate * reward
            self.flux_mat = np.clip(self.flux_mat, 0, self.max_flux)
            self.extended_mat += self.flux_displacements
            self.extended_mat = np.clip(self.extended_mat, self.p_min, self.p_max)
            self.center_mat = np.clip(
                self.center_mat,
                self.p_min,
                self.p_max,
            )
        self.flux_displacements = self.flux_mat * np.sin(
            self.moments / self.periods * 2 * np.pi
        )
