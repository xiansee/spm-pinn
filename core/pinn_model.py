from typing import Callable

from torch import Tensor, log, nn, sqrt


class DNN(nn.Module):
    """
    Deep neural network.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers

        self.dnn = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
            *[nn.Linear(self.hidden_size, self.hidden_size), nn.Tanh()]
            * num_hidden_layers,
            nn.Linear(self.hidden_size, self.output_size),
            nn.Tanh(),
        )

    def forward(self, X: Tensor) -> Tensor:
        dnn_output = self.dnn(X)

        return dnn_output


class SPM_PINN(nn.Module):
    """
    Single Particle Model with solid diffusion modeled by neural networks.
    """

    def __init__(
        self,
        nn_hidden_size: int,
        nn_num_hidden_layers: int,
        Up: Callable,  # Positive electrode OCP as f(conc) [V]
        Cp_0: float,  # Initial positive electrode Li concentration [mol/m3]
        Cp_max: float,  # Max positive electrode Li concentration [mol/m3]
        Rp: float,  # Positive electrode particle radius [m]
        ep_s: float,  # Positive electrode volume fraction [-]
        Lp: float,  # Positive Electrode thickness [m]
        kp: float,  # Positive electrode reaction rate constant [m^2.5/(mol^0.5.s)]
        Dp: float,  # Positive electrode diffusivity [m2/s]
        Un: Callable,  # Negative electrode OCP as f(conc) [V]
        Cn_0: float,  # Initial negative electrode Li concentration [mol/m3]
        Cn_max: float,  # Max negative electrode Li concentration [mol/m3]
        Rn: float,  # Negative electrode particle radius [m]
        en_s: float,  # Negative electrode volume fraction [-]
        Ln: float,  # Negative Electrode thickness [m]
        kn: float,  # Negative electrode reaction rate constant [m^2.5/(mol^0.5.s)]
        Dn: float,  # Negative electrode diffusivity [m2/s]
        Ce: float,  # Electrolyte Li concentration [mol/m3]
        R_cell: float,  # Cell resistance [ohm m2]):
    ):
        super().__init__()
        self.nn_hidden_size = nn_hidden_size
        self.nn_num_hidden_layers = nn_num_hidden_layers

        self.Up, self.Un = Up, Un

        self.Cp_0 = Cp_0
        self.Cp_max = Cp_max
        self.Rp = Rp
        self.ep_s = ep_s
        self.Lp = Lp
        self.kp = kp
        self.ap = 3 * self.ep_s / self.Rp
        self.Dp = Dp

        self.Cn_0 = Cn_0
        self.Cn_max = Cn_max
        self.Rn = Rn
        self.en_s = en_s
        self.Ln = Ln
        self.kn = kn
        self.an = 3 * self.en_s / self.Rn
        self.Dn = Dn

        self.Ce = Ce
        self.R_cell = R_cell

        self.T = 298  # Temperature [K]
        self.F = 96485.33  # Faraday constant [C/mol]
        self.R = 8.314  # Universal gas constant [J/mol.K]

        self._init_solid_diffusion_models()

    def _init_solid_diffusion_models(self):
        # Positive electrode particle concentration NN
        self.Cp_dnn = DNN(
            input_size=2,
            hidden_size=self.nn_hidden_size,
            output_size=1,
            num_hidden_layers=self.nn_num_hidden_layers,
        )

        # Negative electrode particle concentration NN
        self.Cn_dnn = DNN(
            input_size=2,
            hidden_size=self.nn_hidden_size,
            output_size=1,
            num_hidden_layers=self.nn_num_hidden_layers,
        )

    def forward(self, I: Tensor, Xp: Tensor, Xn: Tensor, N_t: int) -> Tensor:
        jp = -I / (self.F * self.ap * self.Lp * self.Dp)
        jn = I / (self.F * self.an * self.Ln * self.Dn)

        Cp = self.Cp_dnn(Xp)
        Cp = self.unnormalize_data(Cp, max_value=self.Cp_max)
        Cp_surf = Cp[-N_t:]

        Cn = self.Cn_dnn(Xn)
        Cn = self.unnormalize_data(Cn, max_value=self.Cn_max)
        Cn_surf = Cn[-N_t:]

        mp = I / (
            self.F
            * self.kp
            * self.Lp
            * self.ap
            * sqrt(self.Cp_max - Cp_surf)
            * sqrt(Cp_surf)
            * self.Ce**0.5
        )
        mn = I / (
            self.F
            * self.kn
            * self.Ln
            * self.an
            * sqrt(self.Cn_max - Cn_surf)
            * sqrt(Cn_surf)
            * self.Ce**0.5
        )

        kinetics_const = 2 * self.R * self.T / self.F
        V = (
            self.Up(Cp_surf)
            - self.Un(Cn_surf)
            + kinetics_const * log((sqrt(mp**2 + 4) + mp) / 2)
            + kinetics_const * log((sqrt(mn**2 + 4) + mn) / 2)
            + I * self.R_cell
        )

        return V, Cp, Cn, (jp, jn)

    def unnormalize_data(self, C: Tensor, max_value: float):
        return (C + 1) / 2 * max_value
