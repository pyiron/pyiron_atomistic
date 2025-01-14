from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class CalculateMolecularDynamics:
    temperature: float
    pressure: Optional[Union[float, List[float]]]
    n_ionic_steps: int
    time_step: float
    n_print: int
    temperature_damping_timescale: float
    pressure_damping_timescale: float
    seed: Optional[int]
    tloop: Optional[float]
    initial_temperature: Optional[float]
    langevin: bool
    delta_temp: Optional[float]
    delta_press: Optional[float]


@dataclass
class CalculateMinimize:
    ionic_energy_tolerance: float
    ionic_force_tolerance: float
    e_tol: Optional[float]
    f_tol: Optional[float]
    max_iter: int
    pressure: Optional[Union[float, List[float]]]
    n_print: int
    style: str
