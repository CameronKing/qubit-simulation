# A class that implements state evolution for
# the purpose of calculating infidelities using Choi-Jamilkowski
# matrices

import math
from typing import Literal

import numpy as np
import scipy.linalg as LA


class CJ(object):
    """Return a Choi-Jamilkowski matrix after a given amount
    of state evolution.

    This is equivalent to the chi-matrix for the evolution
    """

    def __init__(
        self,
        indices: np.ndarray,
        hamiltonian: np.ndarray,
        noise_hamiltonian: np.ndarray,
        noise_type: Literal["quasistatic", None] = None,
    ):
        """
        Initialize a Choi-Jamilkowski instance with the subspace
        of interest given by indices, and the kernel of the unitary
        evolution, given by hamiltonian (units: angular GHz).

        If non-Hamiltonian evolution is needed go elsewhere.
        """
        dim = hamiltonian.shape[0]
        norm = 1.0 / float(len(indices))
        converted_indices = [(int(dim) + 1) * x for x in indices]
        chi0 = np.zeros((dim**2, dim**2), dtype=complex)
        chi0[np.ix_(converted_indices, converted_indices)] = norm
        self.chi0 = chi0
        self.noise_type = noise_type
        if noise_type == "quasistatic":
            shifted_hamiltonian = hamiltonian + noise_hamiltonian
            shifted_energies = LA.eigh(shifted_hamiltonian)[0]
            shifted_hamiltonian = np.diag(shifted_energies)
            noise_hamiltonian = np.zeros((dim, dim))
            self.kernel = np.kron(np.identity(dim), shifted_hamiltonian)
        else:
            self.kernel = np.kron(np.identity(dim), hamiltonian)
        self.noise = np.kron(np.identity(dim), noise_hamiltonian)
        self.rot_basis = np.kron(np.identity(dim), hamiltonian)

    def chi_final(self, tfinal: float) -> np.ndarray:
        """Using the kernel given in initialition, find the final chi_matrix"""
        if tfinal == 0.0:
            return self.chi0
        else:
            unitary = LA.expm(-1j * tfinal * (self.kernel + self.noise))
            return unitary @ self.chi0 @ unitary.conj().T

    def chi_final_RF(self, tfinal: float) -> np.ndarray:
        """Find the chi_matrix in the rotating frame defined by the deliberate
        rotation
        """
        if tfinal == 0.0:
            return self.chi0
        else:
            unitary_rotation = LA.expm(1j * tfinal * self.rot_basis)
            if self.noise_type == "quasistatic":
                return (
                    unitary_rotation
                    @ self.chi_final(tfinal)
                    @ unitary_rotation.conj().T
                )
            else:
                mod_interaction = (
                    unitary_rotation @ self.noise @ unitary_rotation.conj().T
                )
                unitary_operation = LA.expm(-1j * tfinal * mod_interaction)
                return unitary_operation @ self.chi0 @ unitary_operation.conj().T

    def fidelity(self, tfinal: float) -> float:
        """Calculate the process fidelity.

        Uses: F = tr(chi_{ideal} chi_{actual})

        Parameters
        ----------
        tfinal
            Time of the simulation
            Units: ns

        Returns
        -------
        float
            process fidelity
        """
        noisy_chi = self.chi_final_RF(tfinal)
        return fidelity(self.chi0, noisy_chi)


def fidelity(chi_ideal: np.ndarray, chi_actual: np.ndarray) -> float:
    """Calculate the process fidelity.

    Uses: F = tr(chi_{ideal} chi_{actual})

    Parameters
    ----------
    chi_ideal : (n,n) array
        Ideal process matrix

    chi_actual : (n, n) array
        Actual process matrix

    Returns
    -------
    float
        process fidelity
    """
    chi_ideal = chi_ideal / math.sqrt(np.trace(chi_ideal @ chi_ideal).real)
    chi_actual = chi_actual / math.sqrt(np.trace(chi_actual @ chi_actual).real)
    return np.trace(chi_ideal @ chi_actual).real
