# This is a class structure to deal with evolution
# of quantum systems

import numpy as np
import scipy.linalg as LA


class QuantumState(object):
    """Create a density matrix representing a quantum state."""

    def __init__(self, rho: np.ndarray):
        self.rho = rho

    def evolve_simple(self, kernel: np.ndarray, tfinal: float) -> np.ndarray:
        """Return the evolved state according to the Unitary."""
        unitary = LA.expm(-1j * tfinal * kernel)
        return unitary @ self.rho @ unitary.conj().T

    def norm(self) -> complex:
        return np.sum(np.diag(self.rho))
