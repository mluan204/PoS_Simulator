"""
Gói Mô phỏng PoS
"""

try:
    from .parameters import Parameters, Distribution, PoS, NewEntry, SType
    from .simulator import simulate, simulate_verbose, run_experiment
    from .utils import (
        gini, generate_peers, consensus, 
        weighted_consensus, opposite_weighted_consensus, gini_stabilized_consensus
    )
except ImportError:
    # Khi chạy như script, sử dụng absolute imports
    from parameters import Parameters, Distribution, PoS, NewEntry, SType
    from simulator import simulate, simulate_verbose, run_experiment
    from utils import (
        gini, generate_peers, consensus, 
        weighted_consensus, opposite_weighted_consensus, gini_stabilized_consensus
    )

__version__ = "1.0.0"
__author__ = "Python Port of Julia PoS Simulator"

__all__ = [
    # Các lớp và enum cốt lõi
    'Parameters', 'Distribution', 'PoS', 'NewEntry', 'SType',
    
    # Các hàm mô phỏng
    'simulate', 'simulate_verbose', 'run_experiment',
    
    # Các hàm tiện ích
    'gini', 'generate_peers', 'consensus',
    'weighted_consensus', 'opposite_weighted_consensus', 'gini_stabilized_consensus'
]