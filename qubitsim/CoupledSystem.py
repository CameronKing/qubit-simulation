# A class to describe coupled qubit-resonator systems

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
        return None
        