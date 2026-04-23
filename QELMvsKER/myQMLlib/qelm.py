import numpy as np

class QuantumExtremeLearningMachine:
    """
    Implements a Quantum Extreme Learning Machine (QELM) for regression tasks.
    The QELM to maps classical data into a high-dimensional feature space,
    and then appliesa measurement and a linear regression in the measurement outcomes.
    """
    def __init__(self,
                 isometry: np.ndarray,
                 povm: List[np.ndarray],
                 bipartite_dims: Tuple[int, int],
                 keep_subsystem: int = 1,
                 num_shots: int = None):
        """
        Args:
            isometry
        """
        