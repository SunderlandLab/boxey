import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from .abstractions import BoxeyModel
from dataclasses import dataclass
from typing import List, Dict

def display_processes(model: BoxeyModel, process_names: List[str]=[]):
    """Show the processes (rates) composing the model."""
    if len(process_names) < 1:
        process_names = model.processes.keys()

    for name in process_names:
        process = model.processes[name]
        print(f'{process.name}:')
        print(
            f'  {process.timescale} time units {process.compartment_from} -> {process.compartment_to}')
        print(f' reference: {process.reference}')


def get_flux(model: BoxeyModel, t: float, name: str, reservoirs: np.ndarray) -> float:
    """Fluxes associated with named proces at time t given reservoirs."""
    process = model.processes[name]
    rate = process.get_k(t=t)
    mass = reservoirs[model.compartment_indices[process.compartment_from]]
    return mass*rate


def get_fluxes(model: BoxeyModel, t: float, reservoirs: np.ndarray, names: List[str]=[]) -> Dict[str,float]:
    """Fluxes associated with named processes at time t given reservoirs."""
    if len(names) < 1:
        names = model.processes.keys()
    flux_dict = {}
    for name in names:
        flux_dict[name] = get_flux(model, t, name, reservoirs)
    return flux_dict


def eigen_analysis_plot(model: BoxeyModel):
    """Plot eigenvalues and vectors for current model setup.

    """
    model.decompose()
    vecs = model.eigenvectors
    vals = model.eigenvalues

    plt.figure(figsize=(model.N*2, model.N))
    inds = range(model.N)
    for i in inds:
        vec = 0.5*vecs[:, i]
        plt.plot(i+np.real(vec), inds, 'o-', color='k')
        plt.plot(i+np.imag(vec), inds, 'o-', color='g')
        plt.axvline(i, color='gray', linestyle='--')
        plt.axhline(i, color='gray', linestyle='--')
    printvals = []
    for x in -1/vals:
        if x.imag == 0.:
            printvals.append('%.1f' % x.real)
        else:
            printvals.append('%.1f%+.1fi' % (x.real, x.imag))
    plt.xticks(inds, printvals, fontsize=15)
    plt.yticks(inds, model.compartments, fontsize=15)

def plot_perturbation(model: BoxeyModel, tmax: float=4, tmin: float=-2, perturbed_compartment: int=0, 
            perturbation_vector: np.ndarray=None, newfig: bool=True):
    """Make a perturbation diagram for model.
    
    tmax: plot time axis will go out to 10**tmax (default 4)
    tmin: plot time axis will start at 10**tmin (default -2)
    perturbed_compartment: index of single compartment to perturb (default 0)
    perturbation_vector: array of perturbations to each compartment (default None)
    newfig: whether to create a new figure (default True)

    returns: perturbation_timeseries (Ncomp x 1000), times (1x1000)
    """

    model = deepcopy(model)
    model.set_inputs_to_use([])
    if perturbation_vector is None:
        perturb = np.zeros(model.N)
        perturb[perturbed_compartment] = 1
    else:
        perturb = perturbation_vector
    solution = model.run(np.logspace(tmin,tmax,1000),perturb)
    pert = solution.reservoirs
    tpert = solution.times
    pert = pert.T
    plotperts = pert
    
    comps = model.compartments[::-1]
    if newfig:
        plt.figure(figsize=(8.5,5))
    plt.stackplot(tpert,plotperts[::-1,:],labels=comps,edgecolor='k')
    plt.legend(loc='upper left',fontsize=11)
    plt.semilogx()
    xticks = [1e-2,1,100,10000]
    vlines = np.logspace(tmin,tmax,tmax+3)
    yticks = [0.,0.25,0.5,0.75,1.0]
    for ti in vlines:
        plt.axvline(ti,color='k',linestyle='--',lw=1,alpha=0.3)
    for ti in yticks:
        plt.axhline(ti,color='k',linestyle='--',lw=1,alpha=0.3)
    plt.yticks(yticks)
    plt.xlim(10**tmin,10**tmax); plt.ylim(0,1);

    return pert, tpert

@dataclass
class Solution:
    """Solution to BoxeyModel `run` method."""
    model: BoxeyModel # model used for run
    times: np.ndarray # solution times
    reservoirs: np.ndarray # reservoir values at solution times

    def get_flux(self, time: float, name: str):
        """Get flux due to 'name' at time closest to 'time'."""
        closest = np.argmin(np.abs(self.times - time))
        return get_flux(self.model, time, name, self.reservoirs[closest, :])
    
    def display_processes(self, process_names = []):
        """Show the processes (rates) composing the model."""
        display_processes(self.model, process_names=process_names)

    def get_fluxes(self, time: float, names=[]):
        """Fluxes associated with named processes at given time."""
        closest = np.argmin(np.abs(self.times - time))
        return get_fluxes(self.model, time, self.reservoirs[closest, :], names=names)

    def get_reservoir(self, time: float, name: str):
        """Get mass in reservoir `name` at given `time`.
        
        Interpolates linearly if `time` not a solution point."""
        ci = self.model.compartment_indices[name]
        ti = np.argmin(np.abs(time-self.times))
        if time == self.times[ti]:
            return self.reservoirs[ti,ci]
        elif time > self.times[ti]:
            t0 = ti
            t1 = ti+1
        else:
            t0 = ti-1
            t1 = ti
        c0, c1 = self.reservoirs[t0,ci], self.reservoirs[t1,ci]
        dt = t1-t0
        return (1-(time-t0)/dt)*c0 + (1-(t1-time)/dt)*c1

    def get_reservoirs(self, time: float, names: List[str]):
        """Get mass in reservoirs in compartment list `names` at given `time`.
        
        Interpolates linearly if `time` not a solution point.
        Returns reservoirs in order of compartment list, not internal ordering."""
        return np.array([self.get_reservoir(time, name) for name in names])

    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame representation of Solution object."""
        dict_form = {'Year': self.times}
        for i, compartment in enumerate(self.model.compartments):
            dict_form[compartment] = self.reservoirs[:,i]
        return pd.DataFrame.from_dict(dict_form)