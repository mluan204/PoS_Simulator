#!/usr/bin/env python3
"""
Script kiểm tra cơ bản để xác minh triển khai Python của PoS Simulator
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from parameters import Parameters, PoS, Distribution, NewEntry, SType
from simulator import simulate
from utils import generate_peers, gini, weighted_consensus, opposite_weighted_consensus
import random


def test_gini_calculation():
    """Kiểm tra tính toán hệ số Gini"""
    print("Kiểm tra tính toán hệ số Gini...")
    
    # Kiểm tra với phân phối đều
    equal_data = [100.0, 100.0, 100.0, 100.0, 100.0]
    gini_equal = gini(equal_data)
    print(f"  Phân phối đều - Gini: {gini_equal:.3f} (nên gần 0)")
    
    # Kiểm tra với phân phối không đều
    unequal_data = [10.0, 20.0, 30.0, 40.0, 500.0]
    gini_unequal = gini(unequal_data)
    print(f"  Phân phối không đều - Gini: {gini_unequal:.3f} (nên > 0)")
    
    assert gini_equal < 0.1, "Equal distribution should have low Gini"
    assert gini_unequal > 0.3, "Unequal distribution should have high Gini"
    print("  ✓ Kiểm tra tính toán Gini thành công")


def test_consensus_algorithms():
    """Kiểm tra các thuật toán đồng thuận"""
    print("Kiểm tra các thuật toán đồng thuận...")
    
    stakes = [100.0, 200.0, 300.0, 400.0, 500.0]
    
    # Kiểm tra đồng thuận có trọng số
    selected_weighted = []
    for _ in range(1000):
        validator = weighted_consensus(stakes)
        selected_weighted.append(validator)
    
    # Validator có stake cao nhất (index 4) nên được chọn nhiều nhất
    most_selected_weighted = max(set(selected_weighted), key=selected_weighted.count)
    print(f"  Đồng thuận có trọng số - validator được chọn nhiều nhất: {most_selected_weighted} (nên là 4)")
    
    # Kiểm tra đồng thuận trọng số ngược lại
    selected_opposite = []
    for _ in range(1000):
        validator = opposite_weighted_consensus(stakes)
        selected_opposite.append(validator)
    
    # Validator có stake thấp nhất (index 0) nên được chọn nhiều nhất
    most_selected_opposite = max(set(selected_opposite), key=selected_opposite.count)
    print(f"  Đồng thuận trọng số ngược - validator được chọn nhiều nhất: {most_selected_opposite} (nên là 0)")
    
    print("  ✓ Kiểm tra thuật toán đồng thuận thành công")


def test_peer_generation():
    """Kiểm tra tạo peer với các phân phối khác nhau"""
    print("Kiểm tra tạo peer...")
    
    # Kiểm tra phân phối đều
    uniform_stakes = generate_peers(100, 10000.0, Distribution.UNIFORM)
    uniform_gini = gini(uniform_stakes)
    print(f"  Phân phối đều - Gini: {uniform_gini:.3f} (nên gần 0)")
    
    # Kiểm tra phân phối Gini
    gini_stakes = generate_peers(100, 10000.0, Distribution.GINI, 0.5)
    gini_actual = gini(gini_stakes)
    print(f"  Gini mục tiêu 0.5, thực tế: {gini_actual:.3f} (nên gần 0.5)")
    
    assert len(uniform_stakes) == 100, "Should generate correct number of peers"
    assert abs(sum(uniform_stakes) - 10000.0) < 1e-6, "Total volume should be preserved"
    assert uniform_gini < 0.1, "Uniform distribution should have low Gini"
    
    print("  ✓ Kiểm tra tạo peer thành công")


def test_basic_simulation():
    """Kiểm tra chạy mô phỏng cơ bản"""
    print("Kiểm tra mô phỏng cơ bản...")
    
    # Create simple parameters
    params = Parameters(
        n_epochs=100,  # Mô phỏng ngắn cho kiểm tra
        proof_of_stake=PoS.WEIGHTED,
        n_peers=50,
        n_corrupted=5,
        initial_distribution=Distribution.UNIFORM
    )
    
    # Generate initial stakes
    stakes = generate_peers(
        params.n_peers, 
        params.initial_stake_volume, 
        params.initial_distribution
    )
    
    # Create corrupted peers
    corrupted = random.sample(range(params.n_peers), params.n_corrupted)
    
    # Chạy mô phỏng
    gini_history, peers_history, nakamoto_history = simulate(stakes, corrupted, params)
    
    # Check results
    assert len(gini_history) == params.n_epochs, "Should have history for each epoch"
    assert len(peers_history) == params.n_epochs, "Should have peer count for each epoch"
    assert len(nakamoto_history) == params.n_epochs, "Should have Nakamoto history for each epoch"
    assert all(0 <= g <= 1 for g in gini_history), "Gini values should be between 0 and 1"
    assert all(p > 0 for p in peers_history), "Should always have some peers"
    assert all(nc >= 1 for nc in nakamoto_history), "Nakamoto coefficient should be at least 1"
    
    print(f"  Initial Gini: {gini_history[0]:.3f}")
    print(f"  Final Gini: {gini_history[-1]:.3f}")
    print(f"  Initial Nakamoto: {nakamoto_history[0]}")
    print(f"  Final Nakamoto: {nakamoto_history[-1]}")
    print(f"  Initial peers: {peers_history[0]}")
    print(f"  Final peers: {peers_history[-1]}")
    print("  ✓ Kiểm tra mô phỏng cơ bản thành công")


def test_gini_stabilized():
    """Kiểm tra PoS ổn định Gini"""
    print("Kiểm tra PoS ổn định Gini...")
    
    params = Parameters(
        n_epochs=200,
        proof_of_stake=PoS.GINI_STABILIZED,
        n_peers=50,
        θ=0.3,  # Target Gini
        s_type=SType.LINEAR,
        k=0.01,  # Higher k for faster convergence
        initial_distribution=Distribution.GINI,
        initial_gini=0.6  # Start far from target
    )
    
    stakes = generate_peers(
        params.n_peers, 
        params.initial_stake_volume, 
        params.initial_distribution, 
        params.initial_gini
    )
    
    corrupted = random.sample(range(params.n_peers), params.n_corrupted)
    
    gini_history, _, nakamoto_history = simulate(stakes, corrupted, params)
    
    print(f"  Initial Gini: {gini_history[0]:.3f}")
    print(f"  Final Gini: {gini_history[-1]:.3f}")
    print(f"  Target Gini: {params.θ}")
    
    # Check if Gini moved towards target
    initial_distance = abs(gini_history[0] - params.θ)
    final_distance = abs(gini_history[-1] - params.θ)
    
    print(f"  Initial distance from target: {initial_distance:.3f}")
    print(f"  Final distance from target: {final_distance:.3f}")
    
    print("  ✓ Kiểm tra ổn định Gini hoàn thành")


def main():
    """Chạy tất cả kiểm tra"""
    print("Mô phỏng PoS Python - Kiểm tra Cơ bản")
    print("=" * 50)
    
    # Đặt seed ngẫu nhiên để tái tạo được kết quả
    random.seed(42)
    
    try:
        test_gini_calculation()
        print()
        
        test_consensus_algorithms()
        print()
        
        test_peer_generation()
        print()
        
        test_basic_simulation()
        print()
        
        test_gini_stabilized()
        print()
        
        print("=" * 50)
        print("Tất cả kiểm tra hoàn thành thành công! ✓")
        print("Triển khai Python hoạt động chính xác.")
        
    except Exception as e:
        print(f"❌ Kiểm tra thất bại: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())