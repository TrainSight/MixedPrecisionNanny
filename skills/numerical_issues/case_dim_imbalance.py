"""
案例：维度不平衡导致的大数吃小数问题

背景：无人驾驶场景中，x轴（车辆前进方向）数值范围远大于y轴（横向）
在tensor reduce（sum/mean）时，y轴信息被BF16精度"吃掉"
"""
import torch
from torch.cuda.amp import autocast


def demo_dim_imbalance():
    """演示维度不平衡问题"""
    
    # 模拟数据：batch=4, dims=2 (x, y)
    # x轴范围大（-1000~1000），y轴范围小（-1~1）
    data = torch.tensor([
        [1000.0, 0.5],
        [-800.0, -0.3],
        [1200.0, 0.8],
        [-500.0, -0.1],
    ])
    
    print("原始数据：")
    print(data)
    print(f"维度0（x轴）范围: [{data[:,0].min():.2f}, {data[:,0].max():.2f}]")
    print(f"维度1（y轴）范围: [{data[:,1].min():.2f}, {data[:,1].max():.2f}]")
    print(f"比值: {data[:,0].abs().max() / data[:,1].abs().max():.0f}x\n")
    
    # FP32 reduce
    sum_fp32 = data.sum(dim=0)
    print(f"FP32 sum: {sum_fp32}")
    
    # BF16 reduce
    data_bf16 = data.bfloat16()
    sum_bf16 = data_bf16.sum(dim=0).float()
    print(f"BF16 sum: {sum_bf16}")
    
    # 对比误差
    error = (sum_fp32 - sum_bf16).abs()
    print(f"\n绝对误差: {error}")
    print(f"相对误差: {(error / sum_fp32.abs() * 100).round()}%")
    
    # 关键观察：y轴（维度1）误差大
    print("\n问题分析：")
    print("- x轴数值大（1000级），BF16精度足够表示")
    print("- y轴数值小（0.1级），在reduce时被大数影响精度")


def demo_fix_with_fp32():
    """演示修复：在reduce前转为FP32"""
    
    data = torch.tensor([
        [1000.0, 0.5],
        [-800.0, -0.3],
        [1200.0, 0.8],
        [-500.0, -0.1],
    ])
    
    print("\n修复方案：reduce前转FP32")
    
    # 方法1：在reduce前显式转FP32
    data_bf16 = data.bfloat16()
    sum_fixed = data_bf16.float().sum(dim=0)  # 先转FP32再reduce
    
    print(f"FP32 reference: {data.sum(dim=0)}")
    print(f"Fixed BF16:     {sum_fixed}")
    print(f"误差: {(data.sum(dim=0) - sum_fixed).abs().max():.6f}")


def detect_dim_imbalance(tensor, threshold=1e4):
    """
    检测tensor的维度不平衡问题
    
    Args:
        tensor: 输入tensor
        threshold: 维度间最大比值阈值
    
    Returns:
        如果存在不平衡，返回警告信息列表
    """
    warnings = []
    
    for dim in range(tensor.ndim):
        # 按该维度取最大值和最小值
        dim_max = tensor.abs().max(dim=dim, keepdim=True)[0]
        dim_min = tensor.abs()[tensor != 0].min()  # 非零最小值
        
        ratio = dim_max / dim_min
        max_ratio = ratio.max().item()
        
        if max_ratio > threshold:
            warnings.append({
                'dimension': dim,
                'max_ratio': max_ratio,
                'message': f'维度{dim}存在不平衡，比值{max_ratio:.0f}，reduce操作可能丢失精度'
            })
    
    return warnings


if __name__ == "__main__":
    print("=" * 60)
    print("案例：维度不平衡导致的大数吃小数")
    print("=" * 60)
    
    demo_dim_imbalance()
    demo_fix_with_fp32()
    
    # 使用检测函数
    print("\n" + "=" * 60)
    print("自动检测演示")
    print("=" * 60)
    
    test_data = torch.tensor([
        [1000.0, 0.5],
        [-800.0, -0.3],
        [1200.0, 0.8],
        [-500.0, -0.1],
    ])
    
    warnings = detect_dim_imbalance(test_data)
    if warnings:
        for w in warnings:
            print(f"[WARNING] {w['message']}")
    else:
        print("未检测到维度不平衡")
