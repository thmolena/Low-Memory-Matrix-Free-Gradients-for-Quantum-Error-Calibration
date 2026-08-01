"""Matrix-free local Pauli operators.

The character at position ``q`` in a label acts on bit ``q`` of the computational
basis index, with position zero corresponding to the least-significant bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Union

import numpy as np
from numpy.typing import NDArray

ComplexVector = NDArray[np.complex128]


@dataclass(frozen=True)
class PauliString:
    """Tensor product of single-qubit I, X, Y, and Z operators."""

    label: str

    def __post_init__(self) -> None:
        normalized = self.label.upper()
        if not normalized or set(normalized) - {"I", "X", "Y", "Z"}:
            raise ValueError("label must be a nonempty string over I, X, Y, Z")
        object.__setattr__(self, "label", normalized)

    @property
    def n_qubits(self) -> int:
        return len(self.label)

    @property
    def dimension(self) -> int:
        return 1 << self.n_qubits

    def apply(self, vector: ComplexVector) -> ComplexVector:
        """Apply the Pauli string without constructing a dense matrix."""

        value = np.asarray(vector, dtype=np.complex128)
        if value.ndim != 1 or value.size != self.dimension:
            raise ValueError(
                f"expected a vector of length {self.dimension}, got {value.shape}"
            )

        source = np.arange(self.dimension, dtype=np.int64)
        destination = source.copy()
        phase = np.ones(self.dimension, dtype=np.complex128)

        for qubit, operator in enumerate(self.label):
            bit = (source >> qubit) & 1
            if operator == "X":
                destination ^= 1 << qubit
            elif operator == "Y":
                destination ^= 1 << qubit
                phase *= 1j * (1 - 2 * bit)
            elif operator == "Z":
                phase *= 1 - 2 * bit

        output = np.empty_like(value)
        output[destination] = phase * value
        return output

    def dense(self) -> NDArray[np.complex128]:
        """Construct a dense validation matrix; never used by the estimator."""

        identity = np.eye(self.dimension, dtype=np.complex128)
        return np.column_stack([self.apply(identity[:, j]) for j in range(self.dimension)])


class PauliHamiltonian:
    """Linear Hermitian Hamiltonian ``H(theta) = sum_a theta_a P_a``."""

    def __init__(
        self,
        terms: Sequence[Union[PauliString, str]],
        parameters: Iterable[float],
    ) -> None:
        self.terms = tuple(
            term if isinstance(term, PauliString) else PauliString(term)
            for term in terms
        )
        if not self.terms:
            raise ValueError("at least one Pauli term is required")
        if len({term.n_qubits for term in self.terms}) != 1:
            raise ValueError("all Pauli terms must act on the same number of qubits")

        self.parameters = np.asarray(tuple(parameters), dtype=np.float64)
        if self.parameters.shape != (len(self.terms),):
            raise ValueError("one real parameter is required for each Pauli term")

    @property
    def n_qubits(self) -> int:
        return self.terms[0].n_qubits

    @property
    def dimension(self) -> int:
        return self.terms[0].dimension

    def with_parameters(self, parameters: Iterable[float]) -> "PauliHamiltonian":
        return PauliHamiltonian(self.terms, parameters)

    def matvec(self, vector: ComplexVector) -> ComplexVector:
        output = np.zeros(self.dimension, dtype=np.complex128)
        for coefficient, term in zip(self.parameters, self.terms):
            output += coefficient * term.apply(vector)
        return output

    def derivative_matvecs(self) -> tuple:
        """Return parameter-derivative actions ``v -> P_a v``."""

        return tuple(term.apply for term in self.terms)

    def dense(self) -> NDArray[np.complex128]:
        matrix = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        for coefficient, term in zip(self.parameters, self.terms):
            matrix += coefficient * term.dense()
        return matrix


def local_ising_terms(n_qubits: int) -> tuple[PauliString, ...]:
    """Return ``X_i``, ``Z_i``, and nearest-neighbor ``Z_i Z_{i+1}`` terms."""

    if n_qubits < 2:
        raise ValueError("n_qubits must be at least two")

    labels: list[str] = []
    for operator in ("X", "Z"):
        for qubit in range(n_qubits):
            label = ["I"] * n_qubits
            label[qubit] = operator
            labels.append("".join(label))
    for qubit in range(n_qubits - 1):
        label = ["I"] * n_qubits
        label[qubit] = "Z"
        label[qubit + 1] = "Z"
        labels.append("".join(label))
    return tuple(PauliString(label) for label in labels)
