#!/usr/bin/env python3
"""
Test script để so sánh hiệu quả của log weighted consensus trong việc giảm Gini
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from src.utils import (
    weighted_consensus, 
    log_weighted_consensus, 
    gini,
    generate_vector_with_gini
)
from src.parameters import PoS, Parameters, Distribution, NewEntry, SType
from src.simulator import simulate

def compare_consensus_algorithms(stakes: List[float], n_rounds: int = 1000) -> dict:
    """
    So sánh các thuật toán đồng thuận về phân phối lựa chọn validator
    
    Args:
        stakes: Danh sách stake ban đầu
        n_rounds: Số vòng mô phỏng
        
    Returns:
        Dictionary chứa kết quả thống kê
    """
    results = {
        'weighted': np.zeros(len(stakes)),
        'log_weighted': np.zeros(len(stakes))
    }
    
    # Mô phỏng lựa chọn validator
    for _ in range(n_rounds):
        # Weighted consensus chuẩn
        selected_weighted = weighted_consensus(stakes)
        results['weighted'][selected_weighted] += 1
        
        # Log weighted consensus
        selected_log = log_weighted_consensus(stakes)
        results['log_weighted'][selected_log] += 1
    
    # Chuyển thành tỷ lệ phần trăm
    for method in results:
        results[method] = results[method] / n_rounds * 100
    
    return results

def simulate_with_rewards(initial_stakes: List[float], pos_type: PoS, 
                         n_epochs: int = 1000, reward: float = 0.1) -> Tuple[List[float], float]:
    """
    Mô phỏng hệ thống PoS với phần thưởng và tính Gini cuối cùng
    """
    params = Parameters(
        n_epochs=n_epochs,
        proof_of_stake=pos_type,
        reward=reward,
        penalty_percentage=0.1,
        p_fail=0.0,  # Không có validator tham nhũng
        p_join=0.0,  # Không có peer mới tham gia
        p_leave=0.0,  # Không có peer rời đi
        join_amount=NewEntry.NEW_AVERAGE,
        initial_distribution=Distribution.UNIFORM,
        s_type=SType.CONSTANT,
        k=0.1,
        θ=0.3
    )
    
    # Chạy simulation
    gini_history, _ = simulate(initial_stakes.copy(), [], params)
    
    return gini_history, gini_history[-1]

def main():
    """Hàm chính để chạy các test"""
    print("🔬 Testing Log Weighted Consensus Algorithm")
    print("=" * 50)
    
    # Tạo dữ liệu test với Gini cao (bất bình đẳng cao)
    n_peers = 10
    initial_volume = 1000.0
    initial_gini = 0.7  # Gini cao = bất bình đẳng cao
    
    stakes = generate_vector_with_gini(n_peers, initial_volume, initial_gini)
    print(f"📊 Initial stakes: {[f'{s:.2f}' for s in stakes]}")
    print(f"📈 Initial Gini coefficient: {gini(stakes):.3f}")
    print()
    
    # Test 1: So sánh phân phối lựa chọn validator
    print("🧪 Test 1: Comparing validator selection distribution")
    print("-" * 40)
    
    results = compare_consensus_algorithms(stakes, 10000)
    
    print("Validator selection frequency (%):")
    print(f"{'Validator':<10} {'Stake':<10} {'Weighted':<12} {'Log Weighted':<15}")
    print("-" * 50)
    
    for i in range(len(stakes)):
        print(f"{i:<10} {stakes[i]:<10.2f} {results['weighted'][i]:<12.2f} {results['log_weighted'][i]:<15.2f}")
    
    # Tính Gini của phân phối lựa chọn
    gini_weighted = gini(results['weighted'].tolist())
    gini_log_weighted = gini(results['log_weighted'].tolist())
    
    print(f"\n📊 Gini of selection distribution:")
    print(f"   Weighted: {gini_weighted:.3f}")
    print(f"   Log Weighted: {gini_log_weighted:.3f}")
    print(f"   Improvement: {((gini_weighted - gini_log_weighted) / gini_weighted * 100):.1f}%")
    print()
    
    # Test 2: Mô phỏng dài hạn với phần thưởng
    print("🧪 Test 2: Long-term simulation with rewards")
    print("-" * 40)
    
    gini_history_weighted, final_gini_weighted = simulate_with_rewards(
        stakes, PoS.WEIGHTED, 1000, 0.1
    )
    
    gini_history_log, final_gini_log = simulate_with_rewards(
        stakes, PoS.LOG_WEIGHTED, 1000, 0.1
    )
    
    print(f"📈 Final Gini coefficients after 1000 epochs:")
    print(f"   Weighted: {final_gini_weighted:.3f}")
    print(f"   Log Weighted: {final_gini_log:.3f}")
    print(f"   Improvement: {((final_gini_weighted - final_gini_log) / final_gini_weighted * 100):.1f}%")
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Validator selection frequency
    plt.subplot(2, 2, 1)
    x = range(len(stakes))
    width = 0.35
    plt.bar([i - width/2 for i in x], results['weighted'], width, 
            label='Weighted', alpha=0.7)
    plt.bar([i + width/2 for i in x], results['log_weighted'], width, 
            label='Log Weighted', alpha=0.7)
    plt.xlabel('Validator Index')
    plt.ylabel('Selection Frequency (%)')
    plt.title('Validator Selection Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Stake distribution
    plt.subplot(2, 2, 2)
    plt.bar(x, stakes, alpha=0.7, color='orange')
    plt.xlabel('Validator Index')
    plt.ylabel('Stake Amount')
    plt.title('Initial Stake Distribution')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Gini evolution
    plt.subplot(2, 2, 3)
    plt.plot(gini_history_weighted, label='Weighted', linewidth=2)
    plt.plot(gini_history_log, label='Log Weighted', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Gini Coefficient')
    plt.title('Gini Evolution Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Log transformation effect
    plt.subplot(2, 2, 4)
    epsilon = 1e-8
    log_stakes = [np.log(max(stake, epsilon) + 1) for stake in stakes]
    plt.scatter(stakes, log_stakes, alpha=0.7, s=100)
    plt.xlabel('Original Stake')
    plt.ylabel('Log(Stake + 1)')
    plt.title('Log Transformation Effect')
    plt.grid(True, alpha=0.3)
    
    # Thêm đường trend
    z = np.polyfit(stakes, log_stakes, 1)
    p = np.poly1d(z)
    plt.plot(stakes, p(stakes), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('log_weighted_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Charts saved to: log_weighted_comparison.png")
    
    # Test 3: Extreme case với sự khác biệt stake lớn
    print("\n🧪 Test 3: Extreme stake inequality")
    print("-" * 40)
    
    # Tạo case cực đoan: 1 whale và 9 minnows
    extreme_stakes = [1000.0] + [1.0] * 9
    extreme_gini = gini(extreme_stakes)
    
    print(f"📊 Extreme stakes: [1000.0, 1.0, 1.0, ..., 1.0]")
    print(f"📈 Extreme Gini: {extreme_gini:.3f}")
    
    extreme_results = compare_consensus_algorithms(extreme_stakes, 10000)
    
    print(f"\n🐋 Whale (validator 0) selection frequency:")
    print(f"   Weighted: {extreme_results['weighted'][0]:.1f}%")
    print(f"   Log Weighted: {extreme_results['log_weighted'][0]:.1f}%")
    
    whale_reduction = ((extreme_results['weighted'][0] - extreme_results['log_weighted'][0]) / 
                      extreme_results['weighted'][0] * 100)
    print(f"   Whale influence reduction: {whale_reduction:.1f}%")
    
    print("\n✅ Log Weighted Consensus Analysis Complete!")
    print("\n💡 Key Insights:")
    print("   - Log transformation reduces influence of large stakeholders")
    print("   - More equitable validator selection distribution")
    print("   - Potential for better decentralization in PoS networks")
    print("   - Maintains security while improving fairness")

if __name__ == "__main__":
    main()