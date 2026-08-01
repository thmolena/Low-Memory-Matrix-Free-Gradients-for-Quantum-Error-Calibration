import numpy as np

from rc_krylov_qec.pauli import PauliHamiltonian, PauliString


def test_single_qubit_paulis_match_standard_matrices() -> None:
    expected = {
        "I": np.array([[1, 0], [0, 1]], dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    for label, matrix in expected.items():
        np.testing.assert_allclose(PauliString(label).dense(), matrix)


def test_hamiltonian_matvec_matches_dense() -> None:
    terms = [PauliString("XI"), PauliString("IZ"), PauliString("ZZ")]
    hamiltonian = PauliHamiltonian(terms, [0.3, -0.8, 0.2])
    vector = np.array([1, 2j, -0.5, 0.7 - 0.2j], dtype=complex)
    np.testing.assert_allclose(
        hamiltonian.matvec(vector),
        hamiltonian.dense() @ vector,
        atol=1e-13,
    )

