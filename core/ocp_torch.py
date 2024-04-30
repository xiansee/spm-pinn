from typing import Callable

import pybamm
from torch import Tensor, exp, tanh

parameter_values = pybamm.ParameterValues("OKane2022")

Cp_max = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
Cn_max = parameter_values["Maximum concentration in negative electrode [mol.m-3]"]


def get_nmc_ocp(Cs: Tensor) -> Callable:
    """
    NMC-811 open circuit potential as a function of concentration. OCP function obtained
    from PyBaMM:
    https://github.com/pybamm-team/PyBaMM/blob/abc42832370268bc765458b4517a53cafdf8d926/pybamm/input/parameters/lithium_ion/Chen2020.py#L76

    Parameters
    ----------
    Cs : float | ndarray
        Particle surface concentration

    Returns
    -------
    Callable
        Open circuit potential as a function of concentration.
    """

    sto = Cs / Cp_max
    return (
        -0.8090 * sto
        + 4.4875
        - 0.0428 * tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * tanh(15.9308 * (sto - 0.3120))
    )


def get_graphite_ocp(Cs: Tensor):
    """
    Graphite open circuit potential as a function of concentration. OCP function obtained
    from PyBaMM:
    https://github.com/pybamm-team/PyBaMM/blob/abc42832370268bc765458b4517a53cafdf8d926/pybamm/input/parameters/lithium_ion/Prada2013.py#L5

    Parameters
    ----------
    Cs : float | ndarray
        Particle surface concentration

    Returns
    -------
    Callable
        Open circuit potential as a function of concentration.
    """

    sto = Cs / Cn_max
    return (
        1.9793 * exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * tanh(30.4444 * (sto - 0.6103))
    )
