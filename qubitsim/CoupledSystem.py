# A class to describe coupled qubit-resonator systems

import math
import numpy as np

class CoupledSystem:
    def __init__(self, qubit, resonator, coupling):
        """
        Create an object consisting of a coupled qubit-resonator system

        Parameters
        ----------

        qubit : Qubit object

        resonator : resonator object

        coupling : float
          Units: GHz
          Strength of the bare coupling
        
        Returns
        -------
        Coupled system object
        """

        self.qubit = qubit
        self.resonator = resonator
        self.coupling = coupling
        self.dim = qubit.dim + resonator.dim


    def bareHamiltonian(self):
        """
        Return the bare Hamiltonian of the system
        """
        qubitH0 = np.kron(self.qubit.hamiltonian_lab(),
                          np.identity(self.resonator.dim))
        resH0 = np.kron(np.identity(self.qubit.dim), 
                        2*math.pi * self.resonator.omega * self.resonator.number_operator())
        return qubitH0 + resH0


    def couplingHamiltonian(self):
        """
        Return the coupling hamiltonian of the system.
        Here we assume that the resonator is connected via the detuning and 
        we do not make the rotating wave approximation.
        Extensions will be required for other assumptions.
        """
        qubitPart = self.qubit.dipole_operator_qubit()
        resPart = self.resonator.creation_operator() + self.resonator.annihilation_operator()
        return 2*math.pi * self.coupling * np.kron(qubitPart, resPart)


    def couplingHamiltonianRotWave(self):
        """
        Return the coupling hamiltonian of the system.
        Here we assume that the resonator is connected via the detuning and 
        The Rotating Wave Approximation is used.

        This assumes low photon numbers. If any leakage states are also 
        nearly resonant with some photon exchange, they will need to be 
        included. See Y.C. Yang 2017 Supplemental for more details.
        """

        # Need to define the qubit raising and lowering operators
        qubitRaise = np.zeros((self.qubit.dim, self.qubit.dim), dtype=complex)
        qubitRaise[0, 1] = 1
        qubitLower = np.zeros((self.qubit.dim, self.qubit.dim), dtype=complex)
        qubitLower[1, 0] = 1

        # Op 1 and Op 2 are different energy conserving operators
        op1 = np.kron(qubitRaise, self.resonator.annihilation_operator())
        op2 = np.kron(qubitLower, self.resonator.creation_operator())
        return 2*math.pi * self.coupling * (op1 + op2)
        