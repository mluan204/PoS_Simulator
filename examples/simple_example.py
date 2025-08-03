#!/usr/bin/env python3
"""
Ví dụ đơn giản minh họa triển khai Python của PoS Simulator
So sánh 4 thuật toán PoS với 2 metrics: Gini và Nakamoto Coefficient
"""

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json

# Thêm src vào path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from parameters import Parameters, PoS, Distribution, NewEntry
from simulator import simulate
from utils import generate_peers, gini



def save_results_to_json(results, filename):
    """Lưu kết quả thí nghiệm dưới dạng JSON"""
    # Chuyển đổi numpy arrays thành list để có thể serialize
    serializable_results = {}
    for key, value in results.items():
        result_data = {
            'gini_history': value['gini_history'] if isinstance(value['gini_history'], list) else value['gini_history'].tolist(),
            'nakamoto_history': value['nakamoto_history'] if isinstance(value['nakamoto_history'], list) else value['nakamoto_history'].tolist(),
        }
        
        # Add optional fields if they exist
        if 'starting_gini' in value:
            result_data['starting_gini'] = value['starting_gini']
        if 'final_gini' in value:
            result_data['final_gini'] = value['final_gini']
        if 'final_nakamoto' in value:
            result_data['final_nakamoto'] = value['final_nakamoto']
        if 'peers_history' in value:
            result_data['peers_history'] = value['peers_history'] if isinstance(value['peers_history'], list) else value['peers_history'].tolist()
        if 'final_peers' in value:
            result_data['final_peers'] = value['final_peers']
            
        serializable_results[key] = result_data
    
    with open(f'results/{filename}', 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"Dữ liệu đã lưu: results/{filename}")


def run_single_experiment(pos_algorithm, experiment_name, starting_gini=0.3):
    """Chạy một thí nghiệm đơn lẻ với thuật toán PoS được chỉ định"""
    print(f"Chạy {experiment_name}")
    
    # Thiết lập tham số chung
    params = Parameters(
        n_epochs=250000,
        proof_of_stake=pos_algorithm,
        initial_stake_volume=5000.0,
        initial_distribution=Distribution.GINI,
        n_peers=10000,
        n_corrupted=50,
        p_fail=0.5,
        p_join=0.001,
        p_leave=0.001,
        join_amount=NewEntry.NEW_RANDOM,
        penalty_percentage=0.5,
        reward=200.0
    )
    
    # Tạo stake ban đầu
    stakes = generate_peers(
        params.n_peers, 
        params.initial_stake_volume, 
        params.initial_distribution, 
        starting_gini
    )
    
    # Tạo các peer bị tham nhũng
    corrupted = random.sample(range(params.n_peers), params.n_corrupted)
    
    print(f"  Initial Gini: {gini(stakes):.3f}")
    print(f"  Peers: {len(stakes)}, Corrupted: {len(corrupted)}")
    
    # Chạy mô phỏng
    gini_history, peers_history, nakamoto_history = simulate(stakes, corrupted, params)
    
    print(f"  Final Gini: {gini_history[-1]:.3f}")
    print(f"  Final Nakamoto: {nakamoto_history[-1]}")
    print(f"  Final Peers: {peers_history[-1]}")
    
    # Tạo filename cho biểu đồ
    filename = experiment_name.lower().replace(' ', '_').replace(':', '')
    
    # Vẽ biểu đồ 1: Gini Coefficient
    plt.figure(figsize=(12, 8))
    plt.plot(gini_history, linewidth=2, color='blue', alpha=0.8)
    plt.title(f'{experiment_name} - Gini Coefficient', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Gini Coefficient')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{filename}_gini.png', dpi=300, bbox_inches='tight')
    print(f"Biểu đồ Gini đã lưu: results/{filename}_gini.png")
    plt.show()
    
    # Vẽ biểu đồ 2: Nakamoto Coefficient
    plt.figure(figsize=(12, 8))
    plt.plot(nakamoto_history, linewidth=2, color='red', alpha=0.8)
    plt.title(f'{experiment_name} - Nakamoto Coefficient', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Nakamoto Coefficient')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{filename}_nakamoto.png', dpi=300, bbox_inches='tight')
    print(f"Biểu đồ Nakamoto đã lưu: results/{filename}_nakamoto.png")
    plt.show()
    
    # Vẽ biểu đồ 3: Peers Count
    plt.figure(figsize=(12, 8))
    plt.plot(peers_history, linewidth=2, color='green', alpha=0.8)
    plt.title(f'{experiment_name} - Peers Count', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Peers')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{filename}_peers.png', dpi=300, bbox_inches='tight')
    print(f"Biểu đồ Peers Count đã lưu: results/{filename}_peers.png")
    plt.show()
    
    # Lưu dữ liệu
    result = {
        'gini_history': gini_history,
        'nakamoto_history': nakamoto_history,
        'starting_gini': starting_gini
    }
    save_results_to_json({0: result}, f'{filename}_data.json')
    
    return result


def run_experiment_1():
    """Thí nghiệm 1: WEIGHTED PoS"""
    return run_single_experiment(PoS.WEIGHTED, "Experiment 1: WEIGHTED PoS")


def run_experiment_2():
    """Thí nghiệm 2: OPPOSITE_WEIGHTED PoS"""
    return run_single_experiment(PoS.OPPOSITE_WEIGHTED, "Experiment 2: OPPOSITE_WEIGHTED PoS")


def run_experiment_3():
    """Thí nghiệm 3: GINI_STABILIZED PoS"""
    return run_single_experiment(PoS.GINI_STABILIZED, "Experiment 3: GINI_STABILIZED PoS")


def run_experiment_4():
    """Thí nghiệm 4: LOG_WEIGHTED PoS"""
    return run_single_experiment(PoS.LOG_WEIGHTED, "Experiment 4: LOG_WEIGHTED PoS")

def run_experiment_5():
    """Thí nghiệm 5: LOG_WEIGHTED_UNIFORM PoS"""
    return run_single_experiment(PoS.LOG_WEIGHTED_UNIFORM, "Experiment 5: LOG_WEIGHTED_UNIFORM PoS")

def run_comparison_experiment():
    """Thí nghiệm 5: So sánh tất cả 5 thuật toán PoS"""
    print("So sánh tất cả 5 thuật toán PoS")
    print("=" * 50)
    
    # Tham số chung cho tất cả algorithms
    base_params = {
        'n_epochs': 250000,
        'initial_stake_volume': 5000.0,
        'initial_distribution': Distribution.GINI,
        'n_peers': 10000,
        'n_corrupted': 50,
        'initial_gini': 0.3,
        'p_fail': 0.5,
        'p_join': 0.001,
        'p_leave': 0.001,
        'join_amount': NewEntry.NEW_RANDOM,
        'penalty_percentage': 0.5,
        'reward': 20.0
    }

        #     n_epochs=20000,
        # proof_of_stake=pos_algorithm,
        # initial_stake_volume=50000.0,
        # initial_distribution=Distribution.GINI,
        # n_peers=10000,
        # n_corrupted=50,
        # p_fail=0.5,
        # p_join=0.001,
        # p_leave=0.001,
        # join_amount=NewEntry.NEW_RANDOM,
        # penalty_percentage=0.5,
        # reward=200.0
    
    # Tạo stakes và corrupted peers (sử dụng cùng dữ liệu cho tất cả)
    stakes_original = generate_peers(
        base_params['n_peers'], 
        base_params['initial_stake_volume'],
        base_params['initial_distribution'], 
        base_params['initial_gini']
    )
    corrupted = random.sample(range(base_params['n_peers']), base_params['n_corrupted'])
    
    print(f"Initial Gini coefficient: {gini(stakes_original):.3f}")
    print(f"Number of peers: {len(stakes_original)}")
    print(f"Number of corrupted peers: {len(corrupted)}")
    print()
    
    # Dictionary để lưu kết quả của từng algorithm
    algorithms = {
        'WEIGHTED': PoS.WEIGHTED,
        'OPPOSITE_WEIGHTED': PoS.OPPOSITE_WEIGHTED,
        'GINI_STABILIZED': PoS.GINI_STABILIZED,
        'LOG_WEIGHTED': PoS.LOG_WEIGHTED,
        'LOG_WEIGHTED_UNIFORM': PoS.LOG_WEIGHTED_UNIFORM
    }
    
    results = {}
    colors = {'WEIGHTED': 'blue', 'OPPOSITE_WEIGHTED': 'red', 
              'GINI_STABILIZED': 'green', 'LOG_WEIGHTED': 'purple', 'LOG_WEIGHTED_UNIFORM': 'orange'}
    
    # Chạy simulation cho từng algorithm
    for name, pos_type in algorithms.items():
        print(f"Running {name} simulation...")
        
        params = Parameters(
            proof_of_stake=pos_type,
            **base_params
        )
        stakes = stakes_original.copy()
        
        gini_history, peers_history, nakamoto_history = simulate(
            stakes, corrupted.copy(), params
        )
        
        results[name] = {
            'gini_history': gini_history,
            'nakamoto_history': nakamoto_history,
            'peers_history': peers_history,
            'final_gini': gini_history[-1],
            'final_nakamoto': nakamoto_history[-1],
            'final_peers': peers_history[-1]
        }
        
        print(f"  Final Gini: {gini_history[-1]:.3f}")
        print(f"  Final Nakamoto: {nakamoto_history[-1]}")
        print(f"  Final Peers: {peers_history[-1]}")
    
    # Vẽ biểu đồ 1: Gini Coefficient Comparison
    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        plt.plot(result['gini_history'], label=name, linewidth=2, 
                color=colors[name], alpha=0.8)
    
    plt.title('Gini Coefficient Evolution - All PoS Algorithms', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Gini Coefficient')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/gini_comparison.png', dpi=300, bbox_inches='tight')
    print("Biểu đồ Gini đã lưu: results/gini_comparison.png")
    plt.show()
    
    # Vẽ biểu đồ 2: Nakamoto Coefficient Comparison
    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        plt.plot(result['nakamoto_history'], label=name, linewidth=2, 
                color=colors[name], alpha=0.8)
    
    plt.title('Nakamoto Coefficient Evolution - All PoS Algorithms', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Nakamoto Coefficient')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/nakamoto_comparison.png', dpi=300, bbox_inches='tight')
    print("Biểu đồ Nakamoto đã lưu: results/nakamoto_comparison.png")
    plt.show()
    
    # Vẽ biểu đồ 3: Peers Count Comparison
    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        plt.plot(result['peers_history'], label=name, linewidth=2, 
                color=colors[name], alpha=0.8)
    
    plt.title('Peers Count Evolution - All PoS Algorithms', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Peers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/peers_comparison.png', dpi=300, bbox_inches='tight')
    print("Biểu đồ Peers Count đã lưu: results/peers_comparison.png")
    plt.show()
    
    # Thống kê chi tiết
    print("\nFINAL COMPARISON RESULTS:")
    print("-" * 70)
    print(f"{'Algorithm':<20} {'Final Gini':<12} {'Final Nakamoto':<15} {'Final Peers':<12}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{name:<20} {result['final_gini']:<12.3f} {result['final_nakamoto']:<15} {result['final_peers']:<12}")
    
    # Tìm algorithm tốt nhất cho từng metric
    best_gini = min(results.items(), key=lambda x: x[1]['final_gini'])
    best_nakamoto = max(results.items(), key=lambda x: x[1]['final_nakamoto'])
    
    print(f"\nBest for lowest Gini: {best_gini[0]} ({best_gini[1]['final_gini']:.3f})")
    print(f"Best for highest Nakamoto: {best_nakamoto[0]} ({best_nakamoto[1]['final_nakamoto']})")

    # Lưu dữ liệu
    save_results_to_json(results, 'all_pos_comparison_data.json')
    
    print("\nComparison completed!")
    return results


def main():
    """Chạy thí nghiệm PoS Simulator"""
    print("PoS Simulator - So sánh 5 Thuật toán Proof-of-Stake")
    print("=" * 60)
    
    # Đặt seed ngẫu nhiên để tái tạo được kết quả
    random.seed(42)
    np.random.seed(42)
    
    # Tạo thư mục kết quả
    os.makedirs("results", exist_ok=True)
    print("Thư mục 'results' đã được tạo để lưu biểu đồ")
    
    try:
        while True:
            print("\nChọn thí nghiệm:")
            print("1. Experiment 1: WEIGHTED PoS")
            print("2. Experiment 2: OPPOSITE_WEIGHTED PoS")
            print("3. Experiment 3: GINI_STABILIZED PoS") 
            print("4. Experiment 4: LOG_WEIGHTED PoS")
            print("5. Experiment 5: LOG_WEIGHTED_UNIFORM PoS")
            print("6. So sánh tất cả 5 thuật toán")
            print("7. Thoát")
            
            choice = input("\nNhập lựa chọn (1-7): ").strip()
            
            if choice == "1":
                print("\n" + "=" * 60)
                run_experiment_1()
            elif choice == "2":
                print("\n" + "=" * 60)
                run_experiment_2()
            elif choice == "3":
                print("\n" + "=" * 60)
                run_experiment_3()
            elif choice == "4":
                print("\n" + "=" * 60)
                run_experiment_4()
            elif choice == "5":
                print("\n" + "=" * 60)
                run_experiment_5()
            elif choice == "6":
                print("\n" + "=" * 60)
                run_comparison_experiment()
            elif choice == "7":
                print("Tạm biệt!")
                break
            else:
                print("Lựa chọn không hợp lệ. Vui lòng thử lại.")
        
        print("\n" + "=" * 60)
        print("Tất cả thí nghiệm hoàn thành thành công!")
        
    except Exception as e:
        print(f"Lỗi khi thực thi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()