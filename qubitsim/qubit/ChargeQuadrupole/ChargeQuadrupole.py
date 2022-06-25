# Class definition for the charge quadrupole qubit

import math
from typing import Literal

import numpy as np
import scipy.linalg as LA


class ChargeQuadrupole(object):
    """Return a charge dipole qubit object."""

    def __init__(self, eq: float, delta1: float, delta2: float, ed: float):
        """
        Create an instance of a hybrid qubit. The accessible features
        are ed, stsplitting, delta1, delta2, and dim.

        Parameters
        ----------
        eq
            quadrupolar detuning
            Units: GHz
        delta1
            CL tunnel coupling
            Units: GHz
        delta2
            CR tunnel coupling
            Units: GHz
        ed
            dipolar detuning
            Units: GHz

        """
        __slots__ = (
            "eq",
            "delta1",
            "delta2",
            "ed",
        )
        self.eq = eq
        self.delta1 = delta1
        self.delta2 = delta2
        self.ed = ed
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
        eq = self.eq
        delta1 = self.delta1
        delta2 = self.delta2
        ed = self.ed
        H0 = np.array(
            [
                [
                    0.5 * eq,
                    (delta1 + delta2) / math.sqrt(2),
                    (delta1 - delta2) / math.sqrt(2),
                ],
                [(delta1 + delta2) / math.sqrt(2), -0.5 * eq, ed],
                [(delta1 + delta2) / math.sqrt(2), ed, -0.5 * eq],
            ]
        )
        return 2 * math.pi * H0

    def c_detune_qubit_splitting(self, deviation: complex) -> complex:
        """From the base hamiltonian, find the complex form of the qubit's
        energy resulting from a complex deviation in the detuning.

        Parameters
        ----------
        deviation
            a complex deviation in the dipolar detuning
            Units: GHz

        Returns
        -------
        deviation
            qubit splitting energy
            units: GHz
        """
        Hbase = self.hamiltonian_lab() / (2 * math.pi)
        Hdeviation = deviation * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        evals = np.sort(LA.eigvals(Hbase + Hdeviation))
        return evals[1] - evals[0]

    def qubit_basis(self) -> np.ndarray:
        """Return the qubit eigenbasis sorted according to the convention
        that the first element of every eigenvector must be positive.

        Returns
        -------
        (3, 3) float array
            Matrix whose columns are normalized eigenvectors of the system
            system has been sorted into qubit, leakage order
        """
        evecs = LA.eigh(self.hamiltonian_lab())[1]
        evecs = ChargeQuadrupole.eigvector_phase_sort(evecs)
        evecs[:, [0, 1, 2]] = evecs[:, [0, 2, 1]]
        return evecs

    def energies(self) -> np.ndarray:
        """Calculate the system energies from the given Hamiltonian.

        Returns
        -------
        (3,) array
            array of system energies with the qubit space first
            Units: rad/ns
        """
        energies = LA.eigvalsh(self.hamiltonian_lab())
        return energies[[0, 2, 1]]

    def qubit_splitting(self) -> float:
        """Return the qubit splitting in rad/ns.

        Returns
        -------
        float
            qubit energy splitting
            Units: rad/ns
        """
        evals = self.energies()
        return evals[1] - evals[0]

    def detuning_noise_lab(self, ded: float) -> np.ndarray:
        """Return the noise matrix for detuning noise in rad/ns.

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
        return 2 * math.pi * np.array([[0, 0, 0], [0, 0, ded], [0, ded, 0]])

    def detuning_noise_qubit(self, ded: float) -> np.ndarray:
        """
        Return the noise matrix in the qubit frame.

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
        return self.qubit_basis().T @ self.detuning_noise_lab(ded) @ self.qubit_basis()

    def dipole_operator_qubit(self) -> np.ndarray:
        """Calculate the full dipole operator for the hybrid qubit in
        the qubit basis

        Returns
        -------
        (3, 3) array
            dipole operator
            Units: dimensionless
        """
        base_operator = 0.5 * np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        return self.qubit_basis().T @ base_operator @ self.qubit_basis()

    def splitting_derivative(self, order: Literal[1, 2, 3]):
        """Calculate the nth-derivative of the qubit splitting.

        The derivative methods below rely on complex step derivatives for
        numerical stability.

        Parameters
        ----------
        order : int
            order of the derivative (1st, 2nd, 3rd)

        Returns
        -------
        derivative : float
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


class IdealCQ(ChargeQuadrupole):
    """Create an ideal operating point charge quadrupole qubit object"""

    def __init__(self, eq: float, delta: float):
        """Create an charge quadrupole qubit at an optimal working point.

        Parameters
        ----------
        eq
            Quadrupolar detuning
            Units: GHz
        delta
            matched LC, RC tunnel coupling
            Units: GHz
        """
        self.eq = eq
        self.delta = delta
        super().__init__(eq, delta, delta, 0.0)
