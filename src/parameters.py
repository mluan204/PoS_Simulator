"""
Module tham số cho PoS Simulator
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Distribution(Enum):
    """Các loại phân phối cho việc phân bổ stake ban đầu"""
    UNIFORM = 1
    GINI = 2
    RANDOM = 3


class PoS(Enum):
    """Các loại đồng thuận Proof of Stake"""
    WEIGHTED = 1
    OPPOSITE_WEIGHTED = 2
    GINI_STABILIZED = 3
    LOG_WEIGHTED = 4
    LOG_WEIGHTED_UNIFORM = 5  # Thêm thuật toán mới


class NewEntry(Enum):
    """Các loại số lượng mới khi peer tham gia"""
    NEW_MAX = 0
    NEW_MIN = 1
    NEW_RANDOM = 2
    NEW_AVERAGE = 3


class SType(Enum):
    """Các loại hàm cập nhật cho GiniStabilized PoS"""
    CONSTANT = 0
    LINEAR = 1
    QUADRATIC = 2
    SQRT = 3


@dataclass
class Parameters:
    """
    Lớp tham số chứa tất cả các tham số mô phỏng
    """
    n_epochs: int = 50000
    proof_of_stake: PoS = PoS.WEIGHTED
    initial_stake_volume: float = 10000.0
    initial_distribution: Distribution = Distribution.GINI
    initial_gini: float = 0.3
    n_peers: int = 1000
    n_corrupted: int = 20
    p_fail: float = 0.50
    p_join: float = 0.001
    p_leave: float = 0.001
    join_amount: NewEntry = NewEntry.NEW_RANDOM
    penalty_percentage: float = 0.50
    θ: float = 0.3  # theta - hệ số Gini mục tiêu
    s_type: SType = SType.LINEAR
    k: float = 0.001
    reward: float = 10.0
    
    def __post_init__(self):
        """Xác thực sau khi khởi tạo"""
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive")
        if not 0 <= self.initial_gini <= 1:
            raise ValueError("initial_gini must be between 0 and 1")
        if self.n_peers <= 0:
            raise ValueError("n_peers must be positive")
        if self.n_corrupted < 0:
            raise ValueError("n_corrupted must be non-negative")
        if not 0 <= self.p_fail <= 1:
            raise ValueError("p_fail must be between 0 and 1")
        if not 0 <= self.p_join <= 1:
            raise ValueError("p_join must be between 0 and 1")
        if not 0 <= self.p_leave <= 1:
            raise ValueError("p_leave must be between 0 and 1")
        if not 0 <= self.penalty_percentage <= 1:
            raise ValueError("penalty_percentage must be between 0 and 1")
        if not 0 <= self.θ <= 1:
            raise ValueError("θ (theta) must be between 0 and 1")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.reward <= 0:
            raise ValueError("reward must be positive")