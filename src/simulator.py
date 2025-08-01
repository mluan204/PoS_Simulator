"""
Module mô phỏng chính cho PoS Simulator
Tương đương với Simulator.jl từ phiên bản Julia gốc
"""

from typing import List, Tuple
import copy
try:
    from tqdm import tqdm
except ImportError:
    # Phương án dự phòng nếu tqdm không có sẵn
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from .parameters import Parameters, PoS, SType
    from .utils import (
        gini, consensus, d, lerp, try_to_join, try_to_leave,
        nakamoto_coefficient, decentralization_score
    )
except ImportError:
    # Khi chạy như script, sử dụng absolute imports
    from parameters import Parameters, PoS, SType
    from utils import (
        gini, consensus, d, lerp, try_to_join, try_to_leave,
        nakamoto_coefficient, decentralization_score
    )


def simulate(stakes: List[float], corrupted: List[int], params: Parameters) -> Tuple[List[float], List[int], List[int]]:
    """
    Chạy mô phỏng PoS
    
    Args:
        stakes: Stake ban đầu cho mỗi peer
        corrupted: Danh sách chỉ số của các peer bị tham nhũng
        params: Tham số mô phỏng
        
    Returns:
        Tuple của (gini_history, n_peers_history, nakamoto_history)
    """
    # Tạo bản sao để tránh sửa đổi dữ liệu gốc
    stakes = copy.deepcopy(stakes)
    corrupted = copy.deepcopy(corrupted)
    
    gini_history = []
    n_peers_history = []
    nakamoto_history = []
    
    percentage_corrupted = len(corrupted) / len(stakes) if stakes else 0
    
    # Khởi tạo t cho GiniStabilized
    t = d(gini(stakes), params.θ)
    
    for i in range(params.n_epochs):
        # Thử thêm/loại bỏ peer
        try_to_join(stakes, corrupted, params.p_join, params.join_amount, percentage_corrupted)
        try_to_leave(stakes, params.p_leave)
        
        # Tính hệ số Gini hiện tại
        g = gini(stakes)
        gini_history.append(g)
        
        # Tính Nakamoto Coefficient hiện tại
        nc = nakamoto_coefficient(stakes)
        nakamoto_history.append(nc)
        
        # Chọn validator dựa trên cơ chế đồng thuận
        if params.proof_of_stake == PoS.GINI_STABILIZED:
            # Tính s dựa trên s_type
            if params.s_type == SType.CONSTANT:
                s = params.k
            elif params.s_type == SType.LINEAR:
                s = abs(g - params.θ) * params.k
            elif params.s_type == SType.QUADRATIC:
                s = (abs(g - params.θ)) ** 2 * params.k
            else:  # SQRT hoặc khác
                s = (abs(g - params.θ)) ** 0.5 * params.k
            
            validator = consensus(params.proof_of_stake, stakes, t)
            t = lerp(t, d(g, params.θ), s)
        else:
            validator = consensus(params.proof_of_stake, stakes)
        
        # Áp dụng phần thưởng/phạt
        if validator in corrupted and __import__('random').random() > 1 - params.p_fail:
            # Validator bị tham nhũng thất bại - áp dụng phạt
            stakes[validator] *= 1 - params.penalty_percentage
        else:
            # Xác thực thành công - áp dụng phần thưởng
            stakes[validator] += params.reward
        
        # Ghi lại số lượng peer
        n_peers_history.append(len(stakes))
    
    return gini_history, n_peers_history, nakamoto_history


def simulate_verbose(stakes: List[float], corrupted: List[int], params: Parameters) -> Tuple[List[float], List[int], List[int]]:
    """
    Chạy mô phỏng PoS với thanh tiến trình
    
    Args:
        stakes: Stake ban đầu cho mỗi peer
        corrupted: Danh sách chỉ số của các peer bị tham nhũng
        params: Tham số mô phỏng
        
    Returns:
        Tuple của (gini_history, n_peers_history, nakamoto_history)
    """
    # Tạo bản sao để tránh sửa đổi dữ liệu gốc
    stakes = copy.deepcopy(stakes)
    corrupted = copy.deepcopy(corrupted)
    
    gini_history = []
    n_peers_history = []
    nakamoto_history = []
    
    percentage_corrupted = len(corrupted) / len(stakes) if stakes else 0
    
    # Khởi tạo t cho GiniStabilized
    t = d(gini(stakes), params.θ)
    
    # Sử dụng tqdm cho thanh tiến trình (tương đương với @showprogress trong Julia)
    for i in tqdm(range(params.n_epochs), desc="Simulating epochs"):
        # Thử thêm/loại bỏ peer
        try_to_join(stakes, corrupted, params.p_join, params.join_amount, percentage_corrupted)
        try_to_leave(stakes, params.p_leave)
        
        # Tính hệ số Gini hiện tại
        g = gini(stakes)
        gini_history.append(g)
        
        # Tính Nakamoto Coefficient hiện tại
        nc = nakamoto_coefficient(stakes)
        nakamoto_history.append(nc)
        
        # Chọn validator dựa trên cơ chế đồng thuận
        if params.proof_of_stake == PoS.GINI_STABILIZED:
            # Tính s dựa trên s_type
            if params.s_type == SType.CONSTANT:
                s = params.k
            elif params.s_type == SType.LINEAR:
                s = abs(g - params.θ) * params.k
            elif params.s_type == SType.QUADRATIC:
                s = (abs(g - params.θ)) ** 2 * params.k
            else:  # SQRT hoặc khác
                s = (abs(g - params.θ)) ** 0.5 * params.k
            
            validator = consensus(params.proof_of_stake, stakes, t)
            t = lerp(t, d(g, params.θ), s)
        else:
            validator = consensus(params.proof_of_stake, stakes)
        
        # Áp dụng phần thưởng/phạt
        if validator in corrupted and __import__('random').random() > 1 - params.p_fail:
            # Validator bị tham nhũng thất bại - áp dụng phạt
            stakes[validator] *= 1 - params.penalty_percentage
        else:
            # Xác thực thành công - áp dụng phần thưởng
            stakes[validator] += params.reward
        
        # Ghi lại số lượng peer
        n_peers_history.append(len(stakes))
    
    return gini_history, n_peers_history, nakamoto_history


# Hàm tiện lợi để chạy một thí nghiệm đơn lẻ
def run_experiment(n_peers: int, initial_volume: float, initial_gini: float, 
                  params: Parameters, verbose: bool = False) -> Tuple[List[float], List[int], List[int]]:
    """
    Chạy một thí nghiệm đơn lẻ với các tham số cho trước
    
    Args:
        n_peers: Số lượng peer ban đầu
        initial_volume: Khối lượng stake ban đầu
        initial_gini: Hệ số Gini ban đầu
        params: Tham số mô phỏng
        verbose: Có hiển thị thanh tiến trình hay không
        
    Returns:
        Tuple của (gini_history, n_peers_history, nakamoto_history)
    """
    from .utils import generate_peers
    import random
    
    # Tạo stake ban đầu
    stakes = generate_peers(n_peers, initial_volume, params.initial_distribution, initial_gini)
    
    # Tạo các peer bị tham nhũng
    corrupted = random.sample(range(n_peers), params.n_corrupted)
    
    # Chạy mô phỏng
    if verbose:
        return simulate_verbose(stakes, corrupted, params)
    else:
        return simulate(stakes, corrupted, params)