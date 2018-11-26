# This script considers a hybrid qubit coupled to a resonator
# with identical frequencies (resonant operation)

import math

import numpy as np
import scipy.linalg as LA

from context import qubitsim
from qubitsim.qubit import HybridQubit as hybrid
from qubitsim.resonator import Resonator as res

class CoupledSystem():
    def __init__(self, operating_point, match_freq, resdim):
        """
        Create an instance of a coupled sweet-spot qubit and resonator
        that are resonant with each other

        Parameters
        ----------
        operating_point : float
          The ratio of detuning to singlet-triplet splitting at which 
          the qubit should be operated
        match_freq : float
          the operating frequency for both the qubit and the resonator
        resdim : int
          Number of dimensions to include in the resonator
        """
        self.res = res(2*math.pi * match_freq, resdim)
        self.qubit = hybrid.SOSSHybrid(operating_point, match_freq)
        return None

    def H0(self):
        """
        Calculate the uncoupled hamiltonian for the system
        """
        resH0 = np.kron(np.identity(self.qubit.dim), 
                        self.res.omega * self.res.number_operator())
        qubitH0 = np.kron(np.diag(self.qubit.energies()),
                          np.identity(self.res.dim))
        return qubitH0 + resH0
