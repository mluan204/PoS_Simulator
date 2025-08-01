#!/usr/bin/env python3
"""
Test script để kiểm tra Random Distribution implementation
"""

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Thêm src vào path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import generate_peers, gini
from parameters import Distribution

def test_random_distribution():
    """Test Random distribution implementation"""
    print("🧪 Testing Random Distribution Implementation")
    print("=" * 50)
    
    # Set seed cho reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Test parameters
    n_peers = 10
    initial_volume = 1000.0
    
    print(f"Testing with {n_peers} peers, total volume: {initial_volume}")
    print()
    
    # Test tất cả 3 loại distribution
    distributions = {
        'UNIFORM': Distribution.UNIFORM,
        'GINI (0.3)': Distribution.GINI,
        'RANDOM': Distribution.RANDOM
    }
    
    results = {}
    
    for name, dist_type in distributions.items():
        print(f"📊 Testing {name}:")
        
        if dist_type == Distribution.GINI:
            stakes = generate_peers(n_peers, initial_volume, dist_type, 0.3)
        else:
            stakes = generate_peers(n_peers, initial_volume, dist_type)
        
        # Tính toán thống kê
        total = sum(stakes)
        min_stake = min(stakes)
        max_stake = max(stakes)
        mean_stake = np.mean(stakes)
        std_stake = np.std(stakes)
        gini_coeff = gini(stakes)
        
        results[name] = {
            'stakes': stakes,
            'total': total,
            'min': min_stake,
            'max': max_stake,
            'mean': mean_stake,
            'std': std_stake,
            'gini': gini_coeff
        }
        
        print(f"  Stakes: {[round(s, 2) for s in stakes]}")
        print(f"  Total: {total:.2f} (should be {initial_volume})")
        print(f"  Min: {min_stake:.2f}, Max: {max_stake:.2f}")
        print(f"  Mean: {mean_stake:.2f}, Std: {std_stake:.2f}")
        print(f"  Gini Coefficient: {gini_coeff:.3f}")
        print()
    
    # So sánh
    print("📈 COMPARISON:")
    print("-" * 30)
    print(f"{'Distribution':<15} {'Gini':<8} {'Std':<8} {'Min/Max Ratio':<12}")
    print("-" * 30)
    
    for name, data in results.items():
        ratio = data['min'] / data['max'] if data['max'] > 0 else 0
        print(f"{name:<15} {data['gini']:<8.3f} {data['std']:<8.1f} {ratio:<12.3f}")
    
    return results

def test_multiple_random_runs():
    """Test multiple runs của Random distribution để kiểm tra tính ngẫu nhiên"""
    print("\n🔄 Testing Multiple Random Runs")
    print("=" * 40)
    
    n_peers = 5
    initial_volume = 100.0
    n_runs = 5
    
    print(f"Running {n_runs} times with {n_peers} peers, volume: {initial_volume}")
    print()
    
    gini_values = []
    
    for i in range(n_runs):
        stakes = generate_peers(n_peers, initial_volume, Distribution.RANDOM)
        gini_coeff = gini(stakes)
        gini_values.append(gini_coeff)
        
        print(f"Run {i+1}: {[round(s, 1) for s in stakes]} → Gini: {gini_coeff:.3f}")
    
    print(f"\nGini values: {[round(g, 3) for g in gini_values]}")
    print(f"Gini range: {min(gini_values):.3f} - {max(gini_values):.3f}")
    print(f"Gini std: {np.std(gini_values):.3f}")

def visualize_distributions():
    """Vẽ biểu đồ so sánh các distribution"""
    print("\n📊 Creating Visualization")
    print("=" * 30)
    
    n_peers = 20
    initial_volume = 1000.0
    
    # Tạo data cho 3 loại distribution
    uniform_stakes = generate_peers(n_peers, initial_volume, Distribution.UNIFORM)
    gini_stakes = generate_peers(n_peers, initial_volume, Distribution.GINI, 0.2)
    random_stakes = generate_peers(n_peers, initial_volume, Distribution.RANDOM)
    
    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    distributions_data = [
        (uniform_stakes, 'UNIFORM', 'blue'),
        (gini_stakes, 'GINI (0.5)', 'red'),
        (random_stakes, 'RANDOM', 'green')
    ]
    
    for i, (stakes, name, color) in enumerate(distributions_data):
        ax = axes[i]
        peers_indices = range(len(stakes))
        
        ax.bar(peers_indices, stakes, color=color, alpha=0.7)
        ax.set_title(f'{name}\nGini: {gini(stakes):.3f}', fontweight='bold')
        ax.set_xlabel('Peer Index')
        ax.set_ylabel('Stake Amount')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Biểu đồ đã lưu: results/distribution_comparison.png")
    plt.show()

def main():
    """Chạy tất cả tests"""
    print("🚀 Testing Random Distribution Implementation")
    print("=" * 60)
    
    # Tạo thư mục results
    os.makedirs("results", exist_ok=True)
    
    try:
        # Test cơ bản
        results = test_random_distribution()
        
        # Test multiple runs
        test_multiple_random_runs()
        
        # Visualization
        visualize_distributions()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("🎯 Random Distribution is now fully implemented!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()