"""
Các hàm tiện ích cho Mô phỏng PoS
"""

import numpy as np
import random
from typing import List, Tuple, Optional
try:
    from .parameters import Distribution, PoS, NewEntry, Parameters
except ImportError:
    # Khi chạy như script, sử dụng absolute imports
    from parameters import Distribution, PoS, NewEntry, Parameters


def gini(data: List[float]) -> float:
    """
    Tính hệ số Gini của một tập dữ liệu cho trước.
    
    Hệ số Gini là một thước đo phân tán thống kê đại diện cho 
    sự bất bình đẳng của một phân phối. Thường được sử dụng để đo 
    bất bình đẳng thu nhập trong kinh tế.
    
    Args:
        data: Danh sách chứa các điểm dữ liệu số.
        
    Returns:
        Hệ số Gini của tập dữ liệu đầu vào. Hệ số này nằm trong khoảng 
        từ 0 (bình đẳng hoàn hảo) đến 1 (bất bình đẳng tối đa).
        
    Examples:
        >>> data = [100.0, 200.0, 300.0, 400.0, 500.0]
        >>> gini(data)  # Output: 0.2
    """
    if not data or len(data) == 0:
        return 0.0
    
    # Tính số lượng điểm dữ liệu
    n = len(data)
    
    # Tính tổng của các điểm dữ liệu
    total = sum(data)
    
    if total == 0:
        return 0.0
    
    # Sắp xếp các điểm dữ liệu theo thứ tự tăng dần
    sorted_data = sorted(data)
    
    # Tính phần trăm tích lũy của dữ liệu đã sắp xếp
    cumulative_percentage = np.cumsum(sorted_data) / total
    
    # Tính đường cong Lorenz
    lorenz_curve = cumulative_percentage - 0.5 * (np.array(sorted_data) / total)
    
    # Tính hệ số Gini
    G = 1 - 2 * np.sum(lorenz_curve) / n
    
    return G


def nakamoto_coefficient(data: List[float], threshold: float = 0.51) -> int:
    """
    Tính Nakamoto Coefficient của một tập dữ liệu cho trước.
    
    Nakamoto Coefficient là số lượng thực thể nhỏ nhất cần thiết để 
    kiểm soát một phần trăm nhất định (mặc định 51%) của tổng tài nguyên.
    Đây là một thước đo quan trọng về mức độ phi tập trung trong blockchain.
    
    Args:
        data: Danh sách chứa các điểm dữ liệu số (stake của các validator).
        threshold: Ngưỡng phần trăm cần kiểm soát (mặc định 0.51 = 51%).
        
    Returns:
        Nakamoto Coefficient - số lượng thực thể nhỏ nhất cần thiết để 
        kiểm soát threshold% của tổng tài nguyên.
        
    Examples:
        >>> data = [100.0, 200.0, 300.0, 400.0, 500.0]  # Tổng: 1500
        >>> nakamoto_coefficient(data)  # Cần 2 validator lớn nhất (500+400=900 > 765)
        >>> nakamoto_coefficient(data, 0.33)  # Cần 1 validator (500 > 495)
    """
    if not data or len(data) == 0:
        return 0
    
    # Tính tổng tài nguyên
    total = sum(data)
    
    if total == 0:
        return 0
    
    # Sắp xếp theo thứ tự giảm dần (từ lớn nhất đến nhỏ nhất)
    sorted_data = sorted(data, reverse=True)
    
    # Tính ngưỡng cần kiểm soát
    target_amount = total * threshold
    
    # Tính tổng tích lũy từ lớn nhất
    cumulative_sum = 0
    for i, value in enumerate(sorted_data):
        cumulative_sum += value
        if cumulative_sum >= target_amount:
            return i + 1  # Trả về số lượng thực thể cần thiết
    
    # Trường hợp không đủ tài nguyên (không nên xảy ra)
    return len(data)


def nakamoto_coefficient_analysis(data: List[float]) -> dict:
    """
    Phân tích chi tiết Nakamoto Coefficient với nhiều ngưỡng khác nhau.
    
    Args:
        data: Danh sách chứa các điểm dữ liệu số.
        
    Returns:
        Dictionary chứa Nakamoto Coefficient cho các ngưỡng khác nhau.
        
    Examples:
        >>> data = [100.0, 200.0, 300.0, 400.0, 500.0]
        >>> result = nakamoto_coefficient_analysis(data)
        >>> print(result)
        {'25%': 1, '33%': 1, '50%': 2, '51%': 2, '66%': 3, '75%': 4}
    """
    if not data or len(data) == 0:
        return {}
    
    thresholds = [0.25, 0.33, 0.50, 0.51, 0.66, 0.75]
    results = {}
    
    for threshold in thresholds:
        nc = nakamoto_coefficient(data, threshold)
        results[f"{int(threshold * 100)}%"] = nc
    
    return results


def decentralization_score(data: List[float]) -> float:
    """
    Tính điểm phi tập trung dựa trên Nakamoto Coefficient.
    
    Điểm này được tính bằng cách chuẩn hóa Nakamoto Coefficient 
    theo tổng số thực thể. Điểm cao hơn = phi tập trung hơn.
    
    Args:
        data: Danh sách chứa các điểm dữ liệu số.
        
    Returns:
        Điểm phi tập trung từ 0 đến 1 (1 = hoàn toàn phi tập trung).
        
    Examples:
        >>> data = [100.0, 100.0, 100.0, 100.0, 100.0]  # Phân phối đều
        >>> decentralization_score(data)  # Gần 1.0
        >>> data = [1000.0, 10.0, 10.0, 10.0, 10.0]  # Tập trung
        >>> decentralization_score(data)  # Gần 0.0
    """
    if not data or len(data) == 0:
        return 0.0
    
    n_entities = len(data)
    nc_51 = nakamoto_coefficient(data, 0.51)
    
    # Điểm phi tập trung = (n_entities - nc_51) / (n_entities - 1)
    # Khi nc_51 = 1 (tập trung cao) → score = 0
    # Khi nc_51 = n_entities (phi tập trung hoàn toàn) → score = 1
    # Khi nc_51 = n_entities/2 → score = 0.5
    if n_entities == 1:
        return 0.0
    
    score = (n_entities - nc_51) / (n_entities - 1)
    
    return score


def lerp_vector(a: List[float], b: List[float], l: float) -> List[float]:
    """
    Thực hiện phép nội suy tuyến tính giữa hai vector a và b với một hệ số nội suy l cho trước.
    Nội suy tuyến tính (lerp) tính toán một điểm nằm giữa hai vector a và b, dựa trên một hệ số vô hướng l nằm trong khoảng từ 0 đến 1. Khi l bằng 0, kết quả bằng a; khi l bằng 1, kết quả bằng b.

    Tham số:
        a: Vector bắt đầu.
        b: Vector kết thúc.
        l: Hệ số nội suy nằm trong khoảng từ 0 đến 1.

    Giá trị trả về:
        Một danh sách mới đại diện cho kết quả nội suy tuyến tính giữa a và b với hệ số l.
            
    Ví dụ:
        >>> a = [1.0, 2.0, 3.0]
        >>> b = [4.0, 5.0, 6.0]
        >>> l = 0.5
        >>> lerp_vector(a, b, l)  # Output: [2.5, 3.5, 4.5]
    """
    if len(a) != len(b):
        raise ValueError("Vectors a and b must have the same length")
    
    interpolated_vector = []
    for i in range(len(a)):
        interpolated_vector.append((1 - l) * a[i] + l * b[i])
    
    return interpolated_vector


def lerp(a: float, b: float, l: float) -> float:
    """
    Thực hiện phép nội suy tuyến tính giữa hai giá trị a và b với một hệ số nội suy l cho trước.

    Nội suy tuyến tính (lerp) tính toán một điểm nằm giữa hai điểm a và b, dựa trên một hệ số vô hướng l nằm trong khoảng từ 0 đến 1. Khi l bằng 0, kết quả bằng a; khi l bằng 1, kết quả bằng b.

    Tham số:
        a: Giá trị bắt đầu.
        b: Giá trị kết thúc.
        l: Hệ số nội suy nằm trong khoảng từ 0 đến 1.

    Giá trị trả về:
        Kết quả của phép nội suy tuyến tính giữa a và b với hệ số l.
        
    Ví dụ:
        >>> a = 1.0
        >>> b = 4.0
        >>> l = 0.5
        >>> lerp(a, b, l)  # Output: 2.5
    """
    return (1 - l) * a + l * b


def weighted_consensus(peers: List[float]) -> int:
    """
    Xác định đồng thuận có trọng số giữa một nhóm các nút ngang hàng dựa trên xác suất của họ.
    Hàm này tính toán đồng thuận có trọng số giữa một nhóm các nút ngang hàng, trong đó mỗi nút được đại diện bằng một giá trị xác suất. Các nút có xác suất cao hơn sẽ có ảnh hưởng lớn hơn đến kết quả đồng thuận.

    Tham số:
        peers: Một danh sách chứa số lượng token đã stake của mỗi nút.

    Giá trị trả về:
        Trả về chỉ số của nút được chọn làm kết quả đồng thuận, dựa trên xác suất có trọng số.

    Ví dụ:
    >>> peers = [0.2, 0.3, 0.5]
    >>> weighted_consensus(peers)  # Kết quả: chỉ số của nút được chọn
    """
    if not peers or len(peers) == 0:
        raise ValueError("Peers list cannot be empty")
    
    total = sum(peers)
    if total == 0:
        return random.randint(0, len(peers) - 1)
    
    cumulative_probabilities = np.cumsum(np.array(peers) / total)
    random_number = random.random()
    
    # Tìm chỉ số đầu tiên có xác suất tích lũy >= số ngẫu nhiên
    for i, cum_prob in enumerate(cumulative_probabilities):
        if cum_prob >= random_number:
            return i
    
    # Phương án dự phòng (không nên xảy ra với xác suất tích lũy đúng)
    return len(peers) - 1


def opposite_weighted_consensus(peers: List[float]) -> int:
    """
    Xác định đồng thuận trọng số ngược lại trong nhóm peer dựa trên 
    xác suất của họ.
    
    Hàm này tính toán đồng thuận trọng số ngược lại trong nhóm peer, 
    trong đó mỗi peer được biểu diễn bằng một giá trị xác suất. 
    Các peer có xác suất thấp hơn (ảnh hưởng ngược lại) sẽ có 
    ảnh hưởng lớn hơn đến kết quả đồng thuận.
    
    Args:
        peers: Danh sách chứa số token đã stake của mỗi peer.
        
    Returns:
        Chỉ số của peer được chọn làm đồng thuận ngược lại, dựa trên 
        xác suất trọng số ngược lại.
        
    Examples:
        >>> peers = [0.2, 0.3, 0.5]
        >>> opposite_weighted_consensus(peers)  # Output: index of the selected peer
    """
    if not peers or len(peers) == 0:
        raise ValueError("Peers list cannot be empty")
    
    max_peer = max(peers)
    opposite_peers = [abs(max_peer - peer) for peer in peers]
    
    total = sum(opposite_peers)
    if total == 0:
        return random.randint(0, len(peers) - 1)
    
    cumulative_probabilities = np.cumsum(np.array(opposite_peers) / total)
    random_number = random.random()
    
    # Tìm chỉ số đầu tiên có xác suất tích lũy >= số ngẫu nhiên
    for i, cum_prob in enumerate(cumulative_probabilities):
        if cum_prob >= random_number:
            return i
    
    # Phương án dự phòng (không nên xảy ra với xác suất tích lũy đúng)
    return len(peers) - 1


def gini_stabilized_consensus(peers: List[float], t: float) -> int:
    """
    Xác định đồng thuận ổn định Gini trong nhóm peer dựa trên 
    xác suất của họ, sử dụng nội suy tuyến tính giữa hai phương pháp đồng thuận.
    
    Nó kết hợp hai phương pháp đồng thuận: đồng thuận có trọng số và 
    đồng thuận trọng số ngược lại, sử dụng nội suy tuyến tính dựa trên tham số t.
    
    Args:
        peers: Danh sách chứa số token đã stake của mỗi peer.
        t: Tham số nội suy từ 0 đến 1, xác định trọng số 
           của đồng thuận có trọng số (khi t=0) và đồng thuận trọng số ngược lại (khi t=1).
           
    Returns:
        Chỉ số của peer được chọn làm đồng thuận động.
        
    Examples:
        >>> peers = [0.2, 0.3, 0.5]
        >>> t = 0.5
        >>> gini_stabilized_consensus(peers, t)  # Output: index of the selected peer
    """
    if t == -1:
        raise ValueError("Cannot launch GiniStabilized with t = -1")
    
    if not peers or len(peers) == 0:
        raise ValueError("Peers list cannot be empty")
    
    total = sum(peers)
    if total == 0:
        return random.randint(0, len(peers) - 1)
    
    # Tính xác suất có trọng số
    weighted = np.cumsum(np.array(peers) / total)
    
    # Tính xác suất trọng số ngược lại
    max_peer = max(peers)
    processed_peers = [abs(max_peer - peer) for peer in peers]
    total_processed = sum(processed_peers)
    
    if total_processed == 0:
        opposite_weighted = np.cumsum(np.ones(len(peers)) / len(peers))
    else:
        opposite_weighted = np.cumsum(np.array(processed_peers) / total_processed)
    
    # Nội suy tuyến tính giữa hai phương pháp
    cumulative_probabilities = lerp_vector(opposite_weighted.tolist(), weighted.tolist(), t)
    
    random_number = random.random()
    
    # Tìm chỉ số đầu tiên có xác suất tích lũy >= số ngẫu nhiên
    for i, cum_prob in enumerate(cumulative_probabilities):
        if cum_prob >= random_number:
            return i
    
    # Phương án dự phòng (không nên xảy ra với xác suất tích lũy đúng)
    return len(peers) - 1


def log_weighted_consensus(peers: List[float]) -> int:
    """
    Xác định đồng thuận có trọng số logarit giữa một nhóm các nút ngang hàng.
    
    Hàm này áp dụng logarit tự nhiên lên stake trước khi tính xác suất,
    giúp giảm ảnh hưởng của các validator có stake rất cao và làm giảm
    hệ số Gini của hệ thống.
    
    Args:
        peers: Danh sách chứa số lượng token đã stake của mỗi nút.
        
    Returns:
        Trả về chỉ số của nút được chọn làm kết quả đồng thuận, 
        dựa trên xác suất có trọng số logarit.
        
    Examples:
        >>> peers = [1, 10, 100]
        >>> log_weighted_consensus(peers)  # Giảm ảnh hưởng của peer có stake = 100
    """
    if not peers or len(peers) == 0:
        raise ValueError("Peers list cannot be empty")
    
    # Tránh log(0) bằng cách thêm một giá trị nhỏ
    epsilon = 1e-8
    log_stakes = [np.log(max(stake, epsilon) + 1) for stake in peers]
    
    total = sum(log_stakes)
    if total == 0:
        return random.randint(0, len(peers) - 1)
    
    cumulative_probabilities = np.cumsum(np.array(log_stakes) / total)
    random_number = random.random()
    
    # Tìm chỉ số đầu tiên có xác suất tích lũy >= số ngẫu nhiên
    for i, cum_prob in enumerate(cumulative_probabilities):
        if cum_prob >= random_number:
            return i
    
    # Phương án dự phòng (không nên xảy ra với xác suất tích lũy đúng)
    return len(peers) - 1

def log_weighted_uniform_consensus(peers: List[float], alpha: float = 0.7) -> int:
    """
    Chọn validator theo log-weighted + uniform mix.
    """
    n = len(peers)
    if n == 0:
        raise ValueError("Peers list cannot be empty")
    epsilon = 1e-8
    log_stakes = [np.log(max(stake, epsilon) + 1) for stake in peers]
    total_log = sum(log_stakes)
    if total_log == 0:
        return random.randint(0, n - 1)
    # Xác suất log
    log_probs = np.array(log_stakes) / total_log
    # Xác suất đều
    uniform_probs = np.ones(n) / n
    # Pha trộn
    probs = alpha * log_probs + (1 - alpha) * uniform_probs
    # Tích lũy và chọn
    cumulative_probabilities = np.cumsum(probs)
    random_number = random.random()
    for i, cum_prob in enumerate(cumulative_probabilities):
        if cum_prob >= random_number:
            return i
    return n - 1

def random_consensus(peers: List[float]) -> int:
    """
    Xác định đồng thuận ngẫu nhiên trong nhóm peer.
    
    Hàm này chọn một agent ngẫu nhiên từ nhóm dựa trên phân phối đều.
    
    Args:
        peers: Danh sách chứa số token đã stake của mỗi peer.
        
    Returns:
        Chỉ số của agent được chọn ngẫu nhiên làm đồng thuận.
        
    Examples:
        >>> peers = [0.2, 0.3, 0.5]
        >>> random_consensus(peers)  # Output: index of the randomly selected agent
    """
    if not peers or len(peers) == 0:
        raise ValueError("Peers list cannot be empty")
    
    return random.randint(0, len(peers) - 1)


def constant_reward(total_reward: float, n_epochs: int) -> float:
    """Tính phần thưởng cố định mỗi epoch"""
    return total_reward / n_epochs


def dynamic_reward(total_reward: float, n_epochs: int, current_epoch: int) -> float:
    """Tính phần thưởng động dựa trên epoch hiện tại"""
    return (total_reward / n_epochs) + ((current_epoch / n_epochs) * total_reward)


def generate_peers(n_peers: int, initial_volume: float, distribution_type: Distribution, initial_gini: float = -1.0) -> List[float]:
    """
    Tạo các stake ban đầu của peer dựa trên loại phân phối
    
    Args:
        n_peers: Số lượng peer cần tạo
        initial_volume: Tổng khối lượng stake ban đầu
        distribution_type: Loại phân phối (Uniform, Gini, Random)
        initial_gini: Hệ số Gini ban đầu (cho phân phối Gini)
        
    Returns:
        Danh sách stake ban đầu cho mỗi peer
    """
    if distribution_type == Distribution.UNIFORM:
        return generate_vector_uniform(n_peers, initial_volume)
    elif distribution_type == Distribution.GINI:
        if initial_gini == -1.0:
            print("In order to generate peers with a Gini distribution, call 'generate_peers' with the 'initial_gini' " + 
                  "parameter positive and less or equal to 1. Automatically setting 'initial_gini' equal to 0.3")
            initial_gini = 0.3
        return generate_vector_with_gini(n_peers, initial_volume, initial_gini)
    elif distribution_type == Distribution.RANDOM:
        return generate_vector_random(n_peers, initial_volume)
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")


def consensus(pos: PoS, stakes: List[float], t: float = -1.0) -> int:
    """
    Thực thi thuật toán đồng thuận dựa trên loại PoS
    
    Args:
        pos: Loại Proof of Stake
        stakes: Danh sách stake của peer
        t: Tham số cho đồng thuận GiniStabilized
        
    Returns:
        Chỉ số của validator được chọn
    """
    if pos == PoS.WEIGHTED:
        return weighted_consensus(stakes)
    elif pos == PoS.OPPOSITE_WEIGHTED:
        return opposite_weighted_consensus(stakes)
    elif pos == PoS.GINI_STABILIZED:
        return gini_stabilized_consensus(stakes, t)
    elif pos == PoS.LOG_WEIGHTED:
        return log_weighted_consensus(stakes)
    elif pos == PoS.LOG_WEIGHTED_UNIFORM:
        return log_weighted_uniform_consensus(stakes)
    elif pos == PoS.RANDOM:
        return random_consensus(stakes)
    else:
        raise ValueError(f"Unknown PoS type: {pos}")


def generate_vector_with_gini(n_peers: int, initial_volume: float, gini_coeff: float) -> List[float]:
    """
    Tạo một vector với hệ số Gini cụ thể
    
    Args:
        n_peers: Số lượng peer
        initial_volume: Tổng khối lượng để phân phối
        gini_coeff: Hệ số Gini mục tiêu
        
    Returns:
        Danh sách stake với hệ số Gini đã chỉ định
    """
    def lorenz_curve(x1: float, y1: float, x2: float, y2: float):
        """Tạo hàm đường cong Lorenz"""
        m = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        return lambda x: m * x
    
    max_r = (n_peers - 1) / 2
    r = gini_coeff * max_r
    prop = ((n_peers - 1) / n_peers) * ((max_r - r) / max_r)
    lc = lorenz_curve(0, 0, (n_peers - 1) / n_peers, prop)
    
    # Tạo phân phối tích lũy
    q = [lc(i / n_peers) for i in range(1, n_peers)]
    q.append(1.0)
    
    # Chuyển đổi thành stake thực tế
    cumulative_sum = [i * initial_volume for i in q]
    stakes = [cumulative_sum[0]]
    
    for i in range(1, n_peers):
        stakes.append(cumulative_sum[i] - cumulative_sum[i - 1])
    
    return stakes


def generate_vector_uniform(n: int, volume: float) -> List[float]:
    """
    Tạo phân phối đều của stake
    
    Args:
        n: Số lượng peer
        volume: Tổng khối lượng để phân phối
        
    Returns:
        Danh sách stake bằng nhau
    """
    return [volume / n for _ in range(n)]


def generate_vector_random(n: int, volume: float) -> List[float]:
    """
    Tạo phân phối ngẫu nhiên của stake
    
    Args:
        n: Số lượng peer
        volume: Tổng khối lượng để phân phối
        
    Returns:
        Danh sách stake được phân phối ngẫu nhiên
    """
    if n <= 0:
        return []
    
    if n == 1:
        return [volume]
    
    # Tạo n-1 điểm cắt ngẫu nhiên trong khoảng [0, volume]
    cut_points = sorted([random.uniform(0, volume) for _ in range(n - 1)])
    
    # Tính toán stake cho từng peer dựa trên các điểm cắt
    stakes = []
    
    # Peer đầu tiên: từ 0 đến cut_point đầu tiên
    stakes.append(cut_points[0])
    
    # Các peer ở giữa: từ cut_point trước đến cut_point sau
    for i in range(1, n - 1):
        stakes.append(cut_points[i] - cut_points[i - 1])
    
    # Peer cuối cùng: từ cut_point cuối đến volume
    stakes.append(volume - cut_points[-1])
    
    # Đảm bảo không có stake âm (trong trường hợp rất hiếm)
    stakes = [max(0.0, stake) for stake in stakes]
    
    # Chuẩn hóa để đảm bảo tổng bằng volume chính xác
    total = sum(stakes)
    if total > 0:
        stakes = [stake * volume / total for stake in stakes]
    else:
        # Trường hợp khẩn cấp: trả về phân phối đều
        return generate_vector_uniform(n, volume)
    
    return stakes


def compute_smooth_parameter(current_gini: float, target_gini: float, r: float) -> float:
    """
    Tính tham số làm mịn cho việc ổn định Gini
    
    Args:
        current_gini: Hệ số Gini hiện tại
        target_gini: Hệ số Gini mục tiêu
        r: Hệ số làm mịn
        
    Returns:
        Giá trị tham số làm mịn
    """
    diff = abs(current_gini - target_gini)
    diff = diff * (1 / r)
    
    res = (diff + 0j) ** (1 / 7.0)
    res = res.real * (1 if current_gini >= target_gini else -1)
    
    res = (res / 2) + 0.5
    
    if res > 1.0:
        res = 1.0
    if res < 0.0:
        res = 0.0
    
    return 1 - res


def compute_smooth_parameter2(current_gini: float, target_gini: float, r: float) -> float:
    """
    Cách tính tham số làm mịn thay thế
    
    Args:
        current_gini: Hệ số Gini hiện tại
        target_gini: Hệ số Gini mục tiêu
        r: Hệ số làm mịn
        
    Returns:
        Giá trị tham số làm mịn
    """
    denom = target_gini + r
    if denom == 0:
        return 0.5
    
    res = 0.5 - ((current_gini / denom) - (target_gini / denom)) * (1 / (1 - target_gini / denom))
    
    if res > 1.0:
        res = 1.0
    if res < 0.0:
        res = 0.0
    
    return res


def compute_smooth_parameter3(current_gini: float, target_gini: float) -> float:
    """
    Tính tham số làm mịn đơn giản
    
    Args:
        current_gini: Hệ số Gini hiện tại
        target_gini: Hệ số Gini mục tiêu
        
    Returns:
        Giá trị tham số làm mịn
    """
    if current_gini > target_gini:
        return 0.0
    elif current_gini < target_gini:
        return 1.5
    else:
        return 0.75  # Mặc định cho trường hợp bằng nhau


def d(g: float, θ: float) -> float:
    """
    Tính hàm d cho đồng thuận GiniStabilized
    
    Args:
        g: Hệ số Gini hiện tại
        θ: Hệ số Gini mục tiêu (theta)
        
    Returns:
        Giá trị d
    """
    if g > θ:
        return 0.5
    else:
        return 1.5


def try_to_join(stakes: List[float], corrupted: List[int], p: float, 
                join_amount: NewEntry, percentage_corrupted: float) -> None:
    """
    Thử thêm peer mới vào mạng
    
    Args:
        stakes: Danh sách stake hiện tại (được sửa đổi tại chỗ)
        corrupted: Danh sách chỉ số peer bị tham nhũng (được sửa đổi tại chỗ)
        p: Xác suất tham gia
        join_amount: Loại số lượng cho peer mới
        percentage_corrupted: Phần trăm peer bị tham nhũng
    """
    if random.random() <= p:
        # Thêm peer mới
        if join_amount == NewEntry.NEW_AVERAGE:
            new_stake = sum(stakes) / len(stakes) if stakes else 0
        elif join_amount == NewEntry.NEW_MAX:
            new_stake = max(stakes) if stakes else 0
        elif join_amount == NewEntry.NEW_MIN:
            new_stake = min(stakes) if stakes else 0
        elif join_amount == NewEntry.NEW_RANDOM:
            new_stake = stakes[random.randint(0, len(stakes) - 1)] if stakes else 0
        else:
            new_stake = 0
        
        stakes.append(new_stake)
        
        # Kiểm tra xem peer mới có bị tham nhũng không
        if random.random() <= percentage_corrupted:
            corrupted.append(len(stakes) - 1)
        
        # Gọi đệ quy để thử thêm peer khác
        try_to_join(stakes, corrupted, p, join_amount, percentage_corrupted)


def try_to_leave(stakes: List[float], p: float) -> None:
    """
    Thử loại bỏ peer khỏi mạng
    
    Args:
        stakes: Danh sách stake hiện tại (được sửa đổi tại chỗ)
        p: Xác suất rời khỏi
    """
    if stakes and random.random() <= p:
        # Loại bỏ peer ngẫu nhiên
        index_to_remove = random.randint(0, len(stakes) - 1)
        stakes.pop(index_to_remove)