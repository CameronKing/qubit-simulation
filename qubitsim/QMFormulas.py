# Quantum Mechanics Formulas
# A selection of useful quantum mechanics base formulas
# Cameron King
# University of Wisconsin, Madison

from typing import Literal

import numpy as np


def eigvector_phase_sort(eig_matrix):
    for i in range(eig_matrix.shape[1]):
        if eig_matrix[0, i] < 0:
            eig_matrix[:, i] *= -1
    return eig_matrix


def gaussian(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """Return the value of a normalized gaussian probability distribution
    with mean mu, standard deviation sigma, at the value x"""
    return np.exp(-np.square(x - mean) / (2 * np.square(sigma))) / (
        np.sqrt(2 * np.pi * sigma**2)
    )


def basischange(rho0: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Perform a matrix transformation into the
    basis defined by U. Can also be used for unitary
    transformation of a density matrix rho0"""
    return U @ rho0 @ U.conj().T


def processFidelity(chiIdeal: np.ndarray, chiActual: np.ndarray) -> float:
    """Calculate the process fidelity between
    two process matrices chiIdeal and chiActual.
    chiIdeal and chiActual are not assumed to be unitary
    processes"""
    trace1 = np.real(np.trace(chiIdeal @ chiActual))
    return trace1


def processInfidelity(chiIdeal: np.ndarray, chiActual: np.ndarray) -> float:
    """Calculate the process infidelity between two
    matrices chiIdeal and chiActual. chiIdeal and
    chiActual are not assumed to be unitary processes."""
    return 1 - processFidelity(chiIdeal, chiActual)


def processInfidelityUnitary(chiIdeal: np.ndarray, chiActual: np.ndarray) -> float:
    """Calculate the process fidelity assuming
    unitary processes"""
    return 1 - np.real(np.trace(chiIdeal @ chiActual))


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return the commutator between two equivalently dimensioned
    matrices A and B"""
    return A @ B - B @ A


def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return the anti-commutator between two equivalently dimensioned
    matrices A and B"""
    return A @ B + B @ A


def frobenius_inner(A: np.ndarray, B: np.ndarray) -> float:
    """Return the Frobenius norm between two equivalently dimensioned
    matrices A and B"""
    test1 = np.sqrt(np.abs(np.trace(np.dot(A.conj().T, A.T))))
    test2 = np.sqrt(np.abs(np.trace(np.dot(B.conj().T, B))))
    test3 = np.abs(np.trace(np.dot(A.conj().T, B)))
    return np.sqrt(test3 / (test1 * test2))


def derivative(func, test_point, order: Literal[0, 1, 2, 3]) -> float:
    h = 0.1
    h = 1e-10
    if order == 0:
        return func(test_point)
    elif order == 1:
        test_array = test_point + np.arange(-4 * h, 5 * h, h)
        eval_array = func(test_array)
        coeff_array = np.array(
            [
                1.0 / 280.0,
                -4.0 / 105.0,
                0.2,
                -0.8,
                0.0,
                0.8,
                -0.2,
                -4.0 / 105.0,
                -1.0 / 280.0,
            ]
        )
        return np.dot(coeff_array, eval_array) / h
    elif order == 2:
        test_array = test_point + np.arange(-4 * h, 5 * h, h)
        eval_array = func(test_array)
        coeff_array = np.array(
            [
                -1.0 / 560.0,
                8.0 / 315.0,
                -0.25,
                1.6,
                -205.0 / 72.0,
                1.6,
                -0.2,
                8.0 / 315.0,
                -1.0 / 560.0,
            ]
        )
        return np.dot(coeff_array, eval_array) / h**2
    elif order == 3:
        coeff_array = np.array([-0.5, 1.0, 0.0, -1.0, 0.5])
        return np.dot(coeff_array, eval_array) / h**3
    else:
        raise ValueError("Error in order parameter")
