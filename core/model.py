from typing import Callable

import numpy as np
from fipy import (
    CellVariable,
    DiffusionTerm,
    ExplicitDiffusionTerm,
    SphericalGrid1D,
    TransientTerm,
)
from numpy import log, sqrt

from ocp import get_graphite_ocp, get_nmc_ocp


class SolidDiffusionSolver:
    """
    Finite volume solver of 1D diffusion equation using FiPy.

    """

    t_col = "t [s]"
    j_col = "j [A/m2]"
    r_col = "r [m]"
    c_col = "c [mol/m3]"
    c_surf_col = "c_surf [mol/m3]"

    def __init__(
        self,
        c0: float = 15000.0,
        D: float = 1e-14,
        alpha: float = 0.5,
        R: float = 5e-6,
        nr: int = 20,
    ):
        self.mesh = SphericalGrid1D(nr=nr, Lr=R)
        self.conc = CellVariable(mesh=self.mesh, name=r"$c$", value=c0)
        self.conc.faceGrad.constrain([0.0], self.mesh.facesLeft)

        self.equation = TransientTerm() == DiffusionTerm(
            coeff=alpha * D
        ) + ExplicitDiffusionTerm(coeff=(1.0 - alpha) * D)
        self.reset_data()

    def reset_data(self) -> None:
        self.data = {
            self.t_col: [],
            self.j_col: [],
            self.r_col: [],
            self.c_col: [],
            self.c_surf_col: [],
        }

    def solve(self, time: np.ndarray, current_density: np.ndarray) -> dict:
        self.reset_data()
        dt_array = np.diff(time, prepend=0)

        for index, (dt, j) in enumerate(zip(dt_array, current_density)):
            if dt > 0:
                mass_flux = j
                self.conc.faceGrad.constrain([mass_flux], self.mesh.facesRight)
                self.equation.solve(var=self.conc, dt=dt)

            self.data[self.t_col].append(time[index])
            self.data[self.j_col].append(j)
            self.data[self.r_col].append(np.array(self.mesh.x))
            self.data[self.c_col].append(np.array(self.conc.value))
            self.data[self.c_surf_col].append(self.conc.value[-1])

        return self.data


class SPM:
    """
    Single Particle Model for a lithium-ion battery initialized with default values for
    a LG-M50 cell.

    """

    time_col = "Time [s]"
    current_col = "Current [A/m2]"
    voltage_col = "Voltage [V]"
    rp_col = "Positive Particle Radius [m]"
    cp_col = "Positive Electrode Concentration [mol/m3]"
    cp_surf_col = "Positive Electrode Surface Concentration [mol/m3]"
    rn_col = "Negatibe Particle Radius [m]"
    cn_col = "Negative Electrode Concentration [mol/m3]"
    cn_surf_col = "Negative Electrode Surface Concentration [mol/m3]"

    def __init__(
        self,
        Up: Callable = get_nmc_ocp,  # Positive electrode OCP as f(conc) [V]
        Cp_0: float = 35263,  # Initial positive electrode Li concentration [mol/m3]
        Cp_max: float = 63104,  # Max positive electrode Li concentration [mol/m3]
        Rp: float = 5.22e-6,  # Positive electrode particle radius [m]
        ep_s: float = 0.335,  # Positive electrode volume fraction [-]
        Lp: float = 75.6e-6,  # Positive Electrode thickness [m]
        kp: float = 5e-10,  # Positive electrode reaction rate constant [m^2.5/(mol^0.5.s)]
        Dp: float = 1e-14,  # Positive electrode diffusivity [m2/s]
        Un: Callable = get_graphite_ocp,  # Negative electrode OCP as f(conc) [V]
        Cn_0: float = 15528,  # Initial negative electrode Li concentration [mol/m3]
        Cn_max: float = 33133,  # Max negative electrode Li concentration [mol/m3]
        Rn: float = 5.22e-6,  # Negative electrode particle radius [m]
        en_s: float = 0.335,  # Negative electrode volume fraction [-]
        Ln: float = 75.6e-6,  # Negative Electrode thickness [m]
        kn: float = 5e-10,  # Negative electrode reaction rate constant [m^2.5/(mol^0.5.s)]
        Dn: float = 2e-14,  # Negative electrode diffusivity [m2/s]
        Ce: float = 1000,  # Electrolyte Li concentration [mol/m3]
        R_cell: float = 3.24e-4,  # Cell resistance [ohm m2]
    ):
        self.Up, self.Un = Up, Un

        self.Cp_max = Cp_max
        self.Rp = Rp
        self.ep_s = ep_s
        self.Lp = Lp
        self.kp = kp
        self.ap = 3 * self.ep_s / self.Rp
        self.Dp = Dp

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

        self.Cp_solver = SolidDiffusionSolver(c0=float(Cp_0), D=self.Dp, R=self.Rp)
        self.Cn_solver = SolidDiffusionSolver(c0=float(Cn_0), D=self.Dn, R=self.Rn)

    def solve(
        self,
        duration: float,  # [seconds]
        current_density: float,  # [A/m2]
        delta_t: float = 1,  # [seconds]
    ):
        sim_time = np.linspace(0, duration, int((duration + delta_t) / delta_t))
        jp = np.array(
            [
                -current_density / (self.F * self.ap * self.Lp * self.Dp)
                for _ in sim_time
            ]
        )
        jn = np.array(
            [current_density / (self.F * self.an * self.Ln * self.Dn) for _ in sim_time]
        )

        Cp_data = self.Cp_solver.solve(sim_time, jp)
        Cn_data = self.Cn_solver.solve(sim_time, jn)

        Cp_surf = np.array(Cp_data.get(self.Cp_solver.c_surf_col))
        Cn_surf = np.array(Cn_data.get(self.Cn_solver.c_surf_col))

        mp = current_density / (
            self.F
            * self.kp
            * self.Lp
            * self.ap
            * sqrt(self.Cp_max - Cp_surf)
            * sqrt(Cp_surf)
            * self.Ce**0.5
        )

        mn = current_density / (
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
            + current_density * self.R_cell
        )

        data = {
            self.time_col: sim_time,
            self.current_col: [current_density for _ in sim_time],
            self.voltage_col: V,
            self.rp_col: Cp_data.get(self.Cp_solver.r_col),
            self.cp_col: Cp_data.get(self.Cp_solver.c_col),
            self.cp_surf_col: Cp_surf,
            self.rn_col: Cn_data.get(self.Cn_solver.r_col),
            self.cn_col: Cn_data.get(self.Cn_solver.c_col),
            self.cn_surf_col: Cn_surf,
        }

        return data
