import numpy as np
import numpy.typing as npt


class LRule:
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
        min_flux: npt.NDArray = np.zeros((2, 2)),
        performance_bias: float = 0.00,
        update_rate: float = 0.1,
        dt=0.1,
    ) -> None:
        np.random.seed()

        self.dt = dt
        self.learn_rate = learn_rate
        self.conv_rate = conv_rate
        self.performance_bias = performance_bias
        self.update_rate = update_rate

        # variables for weight centers and weight fluctuations
        self.center_mat = center_mat
        self.flux_displacements = np.zeros(center_mat.shape)
        self.extended_mat = self.center_mat.copy()

        # p_min, p_max are bounds on a parameter's
        self.size = self.center_mat.shape[1]
        self.flux_per_node = self.center_mat.shape[0]
        self.p_min = p_min
        self.p_max = p_max
        self.period_min = period_min
        self.period_max = period_max
        self.max_flux = max_flux
        self.min_flux = min_flux
        self.init_flux = init_flux
        self.flux_mat = self.init_flux.copy()
        self.moments = np.zeros(center_mat.shape)
        self.periods = np.zeros(center_mat.shape)
        self.flux_sign = np.random.binomial(1, 0.5, size=self.center_mat.shape) * 2 - 1

    def iter_moment(self):
        self.moments += self.dt
        new_period = np.where(self.moments > self.periods)
        self.moments[new_period] = 0
        # using uniform distribution for period
        # self.periods[new_period] = np.random.uniform(
        #     low=self.period_min[new_period],
        #     high=self.period_max[new_period],
        #     size=new_period[0].size,
        # )
        # using normal distribution for period
        flux_period_center = (self.period_max[0, 0] + self.period_min[0, 0]) / 2
        dev = (self.period_max[0, 0] - self.period_min[0, 0]) / 4

        self.periods[new_period] = np.random.normal(
            loc=flux_period_center,
            scale=dev,
            size=new_period[0].size,
        )

    def update_weights_with_reward(self, reward, tolerance=0.0, learning=True):
        self.reward = reward - self.performance_bias  # - tolerance
        # update fluctuation size and weight centers
        if abs(tolerance) >= tolerance and learning:
            self.extended_mat = self.center_mat + self.flux_displacements
            self.extended_mat = np.clip(
                self.extended_mat, self.p_min, self.p_max, out=self.extended_mat
            )
            self.center_mat += (
                (self.extended_mat - self.center_mat) * self.learn_rate * self.reward
            )
            self.center_mat = np.clip(
                self.center_mat, self.p_min, self.p_max, out=self.center_mat
            )

            # equation taken from the updating rule in repository
            # is a deviation from the paper description of:
            # \dot(A) = B*R(t)
            # testing the comparison between the two equations
            # https://github.com/InsectRobotics/DynamicSynapseSimplifiedPublic/blob/master/DynamicSynapseSimplified/DynamicSynapseArray2DRandomSin.py
            # within "StepSynapseDynamics"
            # self.Amp *= np.exp(-self.WeightersOscilateDecay*self.ModulatorAmount*dt)
            # self.flux_mat *= np.exp(-self.conv_rate * self.reward)
            self.flux_mat -= self.reward * self.conv_rate
            self.flux_mat = np.clip(
                self.flux_mat,
                a_min=self.min_flux,
                a_max=self.max_flux,
            )
        self.flux_displacements = self.flux_mat * np.sin(
            self.moments / self.periods * 2 * np.pi
        )
