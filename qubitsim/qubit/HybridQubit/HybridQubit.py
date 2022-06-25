# Class definition for the hybrid qubit

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import scipy.linalg as LA
from scipy.optimize import root


class HybridQubit(object):
    """Return a hybrid qubit object."""

    def __init__(self, ed: float, stsplitting: float, delta1: float, delta2: float):
        """
        Create an instance of a hybrid qubit. The accessible features
        are ed, stsplitting, delta1, delta2, and dim.

        Parameters
        ----------
        ed
            dipolar detuning
            Units: GHz
        stsplitting
            singlet-triplet splitting in right dot
            Units: GHz
        delta1
            0-1 tunnel coupling
            Units: GHz
        delta2
            0-2 tunnel coupling
            Units: GHz
        """
        __slots__ = (
            "ed",
            "stsplitting",
            "delta1",
            "delta2",
        )
        self.ed = ed
        self.stsplitting = stsplitting
        self.delta1 = delta1
        self.delta2 = delta2
        self.dim = 3

    def hamiltonian_lab(self) -> np.ndarray:
        """Generate the hamiltonian of the hybrid qubit in the laboratory or
        charge basis.

        Returns
        -------
        H0 : (3, 3)
            Hybrid qubit hamiltonian in the lab frame
            Units: rad/ns
        """
        H0 = np.array(
            [
                [-0.5 * self.ed, 0, self.delta1],
                [0, -0.5 * self.ed + self.stsplitting, -self.delta2],
                [self.delta1, -self.delta2, 0.5 * self.ed],
            ]
        )
        return 2 * math.pi * H0

    def c_detune_qubit_splitting(self, deviation: complex) -> complex:
        """From the base hamiltonian, find the complex form of the qubit's
        energy resulting from a complex deviation in the detuning.

        Parameters
        ----------
        deviation: complex
            a complex deviation in the dipolar detuning
            Units: GHz

        Returns
        -------
        complex
            qubit splitting energy
            units: GHz
        """
        Hbase = self.hamiltonian_lab() / (2 * math.pi)
        Hdeviation = deviation * np.diag([-0.5, -0.5, 0.5])
        evals = np.sort(LA.eigvals(Hbase + Hdeviation))
        return evals[1] - evals[0]

    def qubit_basis(self) -> np.ndarray:
        """Return the qubit eigenbasis sorted according to the convention
        that the first element of every eigenvector must be positive.

        Returns
        -------
        (3, 3) float array
            Matrix whose columns are normalized eigenvectors of the system
        """
        evecs = LA.eigh(self.hamiltonian_lab())[1]
        evecs = HybridQubit.eigvector_phase_sort(evecs)
        return evecs

    def energy_detuning(self, detuning: float) -> np.ndarray:
        """Return the qubit energy as a function of detuning using the
        given values of stsplitting, delta1, delta2.

        Primarily a convenience function for noise derivative calculations.

        Parameters
        ----------
        detuning
            dipolar detuning array
            Units: GHz

        Returns
        -------
        qubit energy
            qubit energies at input detuning values
            Units: rad/ns
        """
        return_array = np.zeros_like(detuning)
        for i in range(len(detuning)):
            H0 = np.array(
                [
                    [-0.5 * detuning[i], 0, self.delta1],
                    [0, -0.5 * detuning[i] + self.stsplitting, -self.delta2],
                    [self.delta1, -self.delta2, 0.5 * detuning[i]],
                ]
            )
            evals = LA.eigvalsh(2 * math.pi * H0)
            return_array[i] = evals[1] - evals[0]
        return return_array

    def energies(self) -> np.ndarray:
        """Calculate the system energies from the given Hamiltonian.

        Returns
        -------
        (3,) float array
            array of system energies
            Units: rad/ns
        """
        return LA.eigvalsh(self.hamiltonian_lab())

    def qubit_splitting(self) -> float:
        """Return the qubit splitting in rad/ns

        Returns
        -------
        float
            qubit energy splitting
            Units: rad/ns
        """
        evals = self.energies()
        return evals[1] - evals[0]

    def detuning_noise_lab(self, ded: float) -> np.ndarray:
        """Return the noise matrix for detuning noise in rad/ns

        Parameters
        ----------
        ded
            diploar detuning
            Units: GHz

        Returns
        -------
        (3, 3) array
            lab frame detuning perturbation
            Units: rad/ns
        """
        return 2 * math.pi * np.diag([-0.5 * ded, -0.5 * ded, 0.5 * ded])

    def detuning_noise_qubit(self, ded: float) -> np.ndarray:
        """Return the noise matrix in the qubit frame.

        Parameters
        ----------
        ded
            dipolar detuning
            Units: GHz

        Returns
        -------
        (3,3) array
            detuning noise matrix in the qubit basis.
            Units: rad/ns
        """
        basis = self.qubit_basis()
        return basis.T @ self.detuning_noise_lab(ded) @ basis

    def dipole_operator_qubit(self) -> np.ndarray:
        """Calculate the full dipole operator for the hybrid qubit in
        the qubit basis.

        Returns
        -------
        (3, 3) array
            dipole operator
            Units: dimensionless
        """
        base_operator = 0.5 * np.diag([-1, -1, 1])
        basis = self.qubit_basis()
        return basis.T @ base_operator @ basis

    def splitting_derivative(self, order: Literal[1, 2, 3]) -> float:
        """Calculate the nth-derivative of the qubit splitting.

        The derivative methods below rely on complex step derivatives for
        numerical stability.

        Parameters
        ----------
        order
            order of the derivative (1st, 2nd, 3rd)

        Returns
        -------
        derivative
            Units: GHz^(order - 1)
        """

        if order == 1:
            step = 1e-10
            c_num = step * (1 + 1.0j) / math.sqrt(2)
            eval1 = self.c_detune_qubit_splitting(c_num)
            eval2 = self.c_detune_qubit_splitting(-c_num)
            return np.imag(eval1 - eval2) / (step * math.sqrt(2))
        elif order == 2:
            step = 1e-8
            c_num = step * (1 + 1.0j) / math.sqrt(2)
            eval1 = self.c_detune_qubit_splitting(c_num)
            eval2 = self.c_detune_qubit_splitting(-c_num)
            return np.imag(eval1 + eval2) / step**2
        elif order == 3:
            step = 1e-8
            c_num = step * (1 + 1.0j)
            eval1 = self.c_detune_qubit_splitting(c_num)
            eval2 = self.c_detune_qubit_splitting(0.5 * c_num)
            eval3 = self.c_detune_qubit_splitting(-0.5 * c_num)
            eval4 = self.c_detune_qubit_splitting(c_num)
            return math.sqrt(2) * np.imag(eval1 - 2 * eval2 + 2 * eval3 - eval4)
        else:
            raise ValueError

    @staticmethod
    def eigvector_phase_sort(eig_matrix: np.ndarray) -> np.ndarray:
        for i in range(eig_matrix.shape[1]):
            if eig_matrix[0, i] < 0:
                eig_matrix[:, i] *= -1
        return eig_matrix


class SOSSHybrid(HybridQubit):
    """Return a hybrid qubit object that is at a second order sweet spot
    resonant at some given frequency
    """

    def __init__(
        self,
        ed_ratio: float,
        matchfreq: float,
        guess: tuple[float, float, float] | None = None,
    ):
        """Create a SOSS hybrid qubit object. It inherits all hybrid qubit
        class methods and has the additional specifications of a particular
        operating point in detuning space and the frequency must match the
        given value.

        Parameters
        ----------
        ed_ratio : float
            ed / stsplitting, the operating point in detuning space
            Units: unitless
        matchfreq : float
            required qubit frequency
            Units: GHz
        guess: (3) float array
            Initial guess for the free qubit parameters
            Order:
                (delta1, delta2, stsplitting)
        """
        __slots__ = (
            "ed_ratio",
            "matchfreq",
        )
        self.ed_ratio = ed_ratio
        self.matchfreq = matchfreq

        if guess is None:
            guess = [0.7 * matchfreq, 0.7 * matchfreq, matchfreq]

        delta1, delta2, stsplitting = SOSSHybrid._find_sweet_spot(
            guess, ed_ratio, matchfreq
        )
        ed = ed_ratio * stsplitting
        super().__init__(ed, stsplitting, delta1, delta2)

    @staticmethod
    def _conditions(
        vector_x: np.ndarray, operating_ratio: float, res_freq: float
    ) -> np.ndarray:
        """This function calculates the required values that
        need to be zero for the SOSS qubit

        Parameters
        ----------
        vector_x
            [delta1, delta2, stsplitting]
            Units: GHz
        operating_ratio
            ed / stsplitting used for operation
            Units: unitless
        res_freq
            qubit frequency to match
            Units: GHz

        Returns
        -------
        (3) array
            [resonance, D1, D2]
                resonance: need to match the given frequency
                Units: GHz
                D1: first derivative of qubit spectrum
                Units: Unitless
                D2: second derivative of qubit spectrum
                Units: GHz^(-1)
        """
        delta1, delta2, stsplitting = vector_x
        ref_qubit = HybridQubit(
            operating_ratio * stsplitting, stsplitting, delta1, delta2
        )
        resonance = res_freq - ref_qubit.qubit_splitting() / (2 * math.pi)
        D1 = ref_qubit.splitting_derivative(1)
        D2 = ref_qubit.splitting_derivative(2)
        return [resonance, D1, D2]

    @staticmethod
    def _find_sweet_spot(
        guess: tuple[float, float, float],
        operating_ratio: tuple[float, float],
        res_freq: float,
    ):
        """
        This method uses the hybrid method and
        scipy.optimize.root to find the values of
        delta1, delta2, and stsplitting that result in a
        second order sweet spot (SOSS).

        Parameters
        ----------
        guess
            a guess for starting values with the form
            [delta1, delta2, stsplitting]
            Units: GHz
        args
            argument tuple of the form (operating_ratio, res_freq)
            operating_ratio: float
                ed / stsplitting for operation
                Units: unitless
            res_freq: float
                the required qubit frequency
                Units: GHz

        Returns
        -------
        tunings: (3)
            values found by scipy.optimize.root to correspond to a SOSS, with
            form [delta1, delta2, stsplitting]
            Units: GHz
        """

        tunings = root(
            SOSSHybrid._conditions,
            guess,
            args=(operating_ratio, res_freq),
            method="hybr",
            tol=1e-8,
            options={"eps": 1e-6, "factor": 10},
        ).x
        return tunings
