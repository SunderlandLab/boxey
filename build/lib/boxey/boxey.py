import numpy as np
import json
from scipy.interpolate import interp1d
from numpy.linalg import eig

from dataclasses import dataclass
from typing import Optional, List, Tuple, Protocol, Dict
from numpy.typing import ArrayLike
from .abstractions import BoxeyModel, BoxeyProcess
from .analysis import Solution
# Model structure

@dataclass
class Process(BoxeyProcess):
    """object to hold individual process information."""

    name: str
    timescale: float
    compartment_from: str
    compartment_to: Optional[str] = None
    reference: Optional[str] = ' '

    def get_k(self, t: float = 0) -> float:
        """First order rate of process."""
        return 1/self.timescale

class Source(Protocol):
    """object to hold individual source information"""
    name: str
    destination: str
    reference: Optional[str] = ' '

    def get_source(self, t: float) -> float:
        """Source magnitude at time t."""

    def get_destination(self) -> str:
        """Destination compartment name."""

    def get_critical_values(self) -> List[float]:
        """Time values at which something needs to change."""

    
class Input(object):
    """object to hold individual input information."""

    name: str
    raw_E: ArrayLike
    compartment_to: str
    raw_t: Optional[ArrayLike] = None
    meta: Optional[Dict] = {}
    reference: Optional[str] = ' '

    def __init__(self, name, E, t, cto, meta={}, ref=' '):
        self.name = name
        self.raw_E = E
        self.raw_t = t
        self.compartment_to = cto
        self.meta = meta
        self.reference = ref

        if t is None:
            self.E_of_t = lambda x: self.raw_E
        else:
            pad_t = np.concatenate(([-4e12], self.raw_t, [4e12]))
            pad_E = np.concatenate((self.raw_E[0:1], self.raw_E,
                                    self.raw_E[-1:]))
            self.E_of_t = interp1d(pad_t, pad_E)

    def get_input(self, t: float = 0) -> float:
        """Get input rate from this source at given time."""
        return self.E_of_t(t)

    def get_raw(self) -> Tuple[ArrayLike, ArrayLike]:
        """Get raw input values."""
        return self.raw_E, self.raw_t


class Model(BoxeyModel):
    """Box model implementation.

    Attributes:
    compartments: names of compartments (list of strings)
    compartment_indices: look-up for compartment index (dict)
    N: number of compartments (int)
    processes: collection of Process objects (dict)
    inputs: collection of Input objects (dict)
    matrix: transfer matrix representation of processes
    inputs_to_use: which inputs to use, by name ('all' or list of strings)

    Methods:
    add_process: add new process to collection
    add_input: add new Input to collection
    build_matrix: translate process collection to matrix form
    get_inputs: look up total inputs at a given time
    run: solve for compartment masses for given times
    run_sources: solve for each named source (Input)
    get_steady_state: solve steady state for given inputs
    as_json: JSON representation of model
    to_json: save as_json to file
    """

    def __init__(self, compartments: List[str]):
        """Initialize model based on listed compartments."""

        self.compartments = compartments
        self.compartment_indices = {
            c: i for i, c in enumerate(compartments)}
        self.N = len(compartments)
        self.processes = {}
        self.inputs = {}
        self.matrix = None
        self.inputs_to_use = 'all'

    def set_inputs_to_use(self, input_list: List[str]):
        """Set whitelist of inputs."""
        self.inputs_to_use = input_list


    def add_process(self, process: Process):
        """Add a process to the model's collection.

        Parameters:
        process: individual process info as process object
        """
        name = process.name
        if name in self.processes.keys():
            print(f'Overwriting process "{name}"...')
        self.processes[name] = process

    def add_input(self, inp: Input):
        """Add an input to the model's collection.

        Parameters:
        inp: individual input info as Input object
        """
        name = inp.name
        if name in self.inputs.keys():
            print(f'Overwriting input "{name}"...')
        self.inputs[name] = inp

    def build_matrix(self, t=0):
        """Translate collection of processes to matrix form."""

        self.matrix = np.zeros((self.N, self.N))
        for name, process in self.processes.items():
            i = self.compartment_indices[process.compartment_from]
            this_k = process.get_k(t)
            # check if flow stays in system:
            if process.compartment_to is not None:
                j = self.compartment_indices[process.compartment_to]
                self.matrix[j, i] += this_k
            self.matrix[i, i] -= this_k

    def get_inputs(self, t: float=0) -> np.ndarray:
        """Get total inputs at given time.

        Parameters:
        t: time (float) [optional]

        Returns: array of total emissions
        """

        if self.inputs_to_use == 'all':
            input_list = self.inputs.keys()
        else:
            input_list = self.inputs_to_use

        E = np.zeros(self.N)
        for input_name in input_list:
            inp = self.inputs[input_name]
            i = self.compartment_indices[inp.compartment_to]
            E[i] += inp.get_input(t)

        return E

    def decompose(self):
        """ Decompose Matrix into eigenvalues/vectors.

        Returns:
            eigenvalues (array): unordered array of eigenvalues
            eigenvectors (array): normalized right eigenvectors
                   ( eigenvectors[:,i] corresponds to eigenvalues[i] )
        """
        eigenvalues, eigenvectors = eig(self.matrix)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.inverse_eigenvectors = np.linalg.inv(self.eigenvectors)
        self.timescales = -1./np.float64(np.real(eigenvalues))
        self.residence_times = -1/np.float64(np.diagonal(self.matrix))
        return eigenvalues, eigenvectors

    def _homogeneous_timestep(self, dt, initial_conditions):
        """Calculate result of unforced evolution over dt.

        initial_conditions in eigenspace
        """
        eigIC = initial_conditions
        in_eig_space = np.exp(self.eigenvalues * dt) * eigIC
        return in_eig_space

    def _inhomogeneous_timestep(self, dt, eigf0, eigf1=None):
        """Calculate result of forcing that changes linearly over dt.

        If eigf1 not given, then constant eigf0 forcing used.
        """

        b = (eigf1-eigf0)/dt
        c = eigf0
        ev = self.eigenvalues
        in_eig_space = (-ev*b*dt - ev*c - b + (ev*c+b)*np.exp(ev*dt))/(ev**2)
        return in_eig_space

    def run(self, times: np.ndarray, initial_conditions=None) -> Solution:
        """Run model to solve for given time points `times`.
        
        times: array of time points to generate solution at
        initial_conditions: array of initial reservoir masses. If None (default), zeros used."""

        if initial_conditions is None:
            initial_conditions = np.zeros(self.N)

        self.decompose()
        all_times = list(times)
        if self.inputs_to_use == 'all':
            input_whitelist = self.inputs.keys()
        else:
            input_whitelist = self.inputs_to_use
        for input_name in input_whitelist:
            thist = self.inputs[input_name].raw_t
            if thist is not None:
                all_times = np.concatenate((all_times, thist))
        needed_times = np.sort(np.unique(all_times))
        needed_times = needed_times[needed_times >= times[0]]
        needed_times = needed_times[needed_times <= times[-1]]
        solution = np.zeros((self.N, len(needed_times)))
        output = np.zeros((self.N, len(times)))
        solution[:, 0] = np.dot(self.inverse_eigenvectors, initial_conditions)
        output[:, 0] = initial_conditions
        j = 1
        for i in range(1, len(needed_times)):
            dt = needed_times[i]-needed_times[i-1]
            Mh = self._homogeneous_timestep(dt, solution[:, i-1])
            Mi = self._inhomogeneous_timestep(dt,
                                             np.dot(self.inverse_eigenvectors, self.get_inputs(
                                                 needed_times[i-1])),
                                             np.dot(self.inverse_eigenvectors, self.get_inputs(needed_times[i])))
            solution[:, i] = Mh + Mi
            if needed_times[i] in times:
                output[:, j] = np.real(np.dot(self.eigenvectors, Mh + Mi))
                j += 1

        return Solution(self, times, output.T)

    def run_rk4(self, tstart: float, tend: float, dt: float=0.01, 
                initial_conditions=None, time_varying_rates=False) -> Solution:
        """Calculate masses in all compartments through time using explicit RK4 scheme.

        Parameters:
        tstart: start time (float)
        tend: end time (float)
        dt: time step (float) [optional]
        initial_conditions: initial mass in each box
                         (None or array) [optional]

        Returns: mass, time
        mass: mass in each compartment through time
                 (2D array of size (Ntime, Ncompartmets))
        time: time axis of solution (1D array of length Ntime)
        """

        nsteps = int((tend - tstart) / dt) + 1
        time = np.linspace(tstart, tend, nsteps)
        M = np.zeros((nsteps, self.N))
        if initial_conditions is None:
            M0 = np.zeros(self.N)
        else:
            M0 = initial_conditions
        M[0, :] = M0
        for i, t in enumerate(time[:-1]): 
            if time_varying_rates:
                self.build_metrix(t)
            K = self.matrix
            k1 = np.dot(K, M[i, :]) + self.get_inputs(t)
            in_mid = self.get_inputs(t+0.5*dt)
            k2 = np.dot(K, M[i, :] + 0.5*k1*dt) + in_mid
            k3 = np.dot(K, M[i, :] + 0.5*k2*dt) + in_mid
            k4 = np.dot(K, M[i, :] + k3*dt) + self.get_inputs(t+dt)
            dM = dt * (k1/6 + k2/3 + k3/3 + k4/6)
            M[i+1, :] = M[i, :] + dM

        return Solution(self, time, M)
                    
    def run_euler(self, tstart: float, tend: float, dt: float=0.01, 
                initial_conditions=None, time_varying_rates=False) -> Solution:
        """Calculate masses in all compartments through time using explicit Euler scheme.

        Parameters:
        tstart: start time (float)
        tend: end time (float)
        dt: time step (float) [optional]
        initial_conditions: initial mass in each box
                         (None or array) [optional]

        Returns: mass, time
        mass: mass in each compartment through time
                 (2D array of size (Ntime, Ncompartmets))
        time: time axis of solution (1D array of length Ntime)
        """

        nsteps = int((tend - tstart) / dt) + 1
        time = np.linspace(tstart, tend, nsteps)
        M = np.zeros((nsteps, self.N))
        if initial_conditions is None:
            M0 = np.zeros(self.N)
        else:
            M0 = initial_conditions
        M[0, :] = M0
        for i, t in enumerate(time[:-1]):
            if time_varying_rates:
                self.build_matrix(t)
            dMdt = np.dot(self.matrix, M[i, :]) + self.get_inputs(t)
            dM = dMdt * dt
            M[i+1, :] = M[i, :] + dM

        return Solution(self, time, M)

    def run_sources(self, sources: List[str], times: np.ndarray, 
        initial_conditions: Dict[str, np.ndarray]={}) -> Dict[str, Solution]:
        """Do a `run` for each given source."""
        previous_selection = self.inputs_to_use[:]
        out = {}
        for source in sources:
            self.set_inputs_to_use([source])
            out[source] = self.run(times, initial_conditions.get(source,None))
        self.set_inputs_to_use(previous_selection)
        return out


    def get_steady_state(self, input_vector: np.ndarray) -> np.ndarray:
        """Calculate the steady state reservoirs associated with given inputs.

        input_vector: vector of inputs to each compartment

        returns: steady state reservoirs
        """
        input_vector = np.array(input_vector)
        return np.linalg.solve(self.matrix, -input_vector)

    def as_json(self):
        """Get JSON represenation of model.

        Will return a JSON representation containing Processes, Inputs, Compartments.
        This representation can be loaded for future use or with other back-ends.
        """
        to_json = {'compartments': self.compartments,
                   'static_processes': [], 'piecewise_sources': []}

        for n, process in sorted(self.processes.items()):
            if process.compartment_to is None:
                to_value = 'out'
            else:
                to_value = process.compartment_to
            to_json['static_processes'].append({'name': process.name, 'timescale': process.timescale,
                                         'destination': to_value,
                                         'origin': process.compartment_from, 'notes': process.reference
                                         })
        for n, inp in sorted(self.inputs.items()):
            if inp.raw_t is None:
                # Eventually will be separate field, but for now:
                to_json['piecewise_sources'].append({'name': inp.name,
                                      'destination': inp.compartment_to,
                                      'magnitudes': [inp.raw_E, inp.raw_E],
                                      'time_axis': [-1e9,1e9],
                                      'notes': inp.reference})
            else:
                to_json['piecewise_sources'].append({'name': inp.name,
                                      'destination': inp.compartment_to,
                                      'magnitudes': list(inp.raw_E),
                                      'time_axis': list(inp.raw_t),
                                      'notes': inp.reference})
        return json.dumps(to_json, indent=4, sort_keys=True)

    def to_json(self, filename: str):
        """Write JSON representation of model to file."""
        with open(filename, 'w') as f:
            f.write(self.as_json())


def model_from_json(filename: str):
    """Construct model from JSON definition file"""
    raise NotImplemented("this hasn't been implemented yet!")
    return


def create_model(compartments: List[str], processes: List[Process]) -> Model:
    """Turn list of compartment names and list of processes into model.

    compartments: list of compartment names
    processes: list of process objects

    returns: created Model object
    """
    model = Model(compartments)

    for process in processes:
        model.add_process(process)
    model.build_matrix()  # model creates matrix

    return model


def add_inputs(model: Model, input_list: List[Input]) -> Model:
    """Add list of Inputs to model.

    model: Model object
    input_list: list of Input objects

    returns: updated Model object
    """

    for inp in input_list:
        model.add_input(inp)

    return model
