"""
案例：NaN/Inf 的产生和传播

常见场景：
1. 前向传播数值溢出
2. Loss计算中的除零
3. 梯度爆炸后的Inf传播
4. Loss Scale调整失败
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


def case1_forward_overflow():
    """案例1：前向传播数值溢出"""
    print("=" * 60)
    print("案例1：前向传播数值溢出")
    print("=" * 60)
    
    # 创建一个会产生大数值的tensor
    x = torch.tensor([1e4, 2e4, 3e4])
    
    print(f"输入: {x}")
    print(f"输入最大值: {x.max().item()}")
    
    # 在FP16中，max是65504
    # 如果我们做一个操作让数值变大...
    with autocast():
        # exp(10000) 会溢出
        try:
            y = torch.exp(x)
            print(f"exp(x): {y}")
            if torch.isinf(y).any():
                print("[ERROR] 出现Inf！")
        except Exception as e:
            print(f"Error: {e}")


def case2_loss_division_by_zero():
    """案例2：Loss计算中的除零"""
    print("\n" + "=" * 60)
    print("案例2：Loss计算中的除零")
    print("=" * 60)
    
    # 模拟一个可能出现0的pred
    pred = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    target = torch.tensor([0.0, 0.5, 1.0])
    
    # 某些loss函数在输入为0时会出问题
    # 比如 log(0) = -inf
    try:
        loss = torch.log(pred)
        print(f"log(pred): {loss}")
        if torch.isinf(loss).any():
            print("[ERROR] 出现Inf！")
    except Exception as e:
        print(f"Error: {e}")


def case3_gradient_explosion():
    """案例3：梯度爆炸导致的Inf"""
    print("\n" + "=" * 60)
    print("案例3：梯度爆炸")
    print("=" * 60)
    
    # 创建一个简单的模型
    model = nn.Linear(10, 1)
    
    # 创建一个梯度会很大的输入
    x = torch.ones(1, 10) * 1000  # 大输入
    target = torch.tensor([1.0])
    
    criterion = nn.MSELoss()
    
    # 不使用AMP，直接FP32计算
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    
    # 检查梯度
    grad_norm = model.weight.grad.norm().item()
    print(f"输入: {x[0,0].item()}")
    print(f"梯度范数: {grad_norm}")
    
    if torch.isinf(model.weight.grad).any():
        print("[ERROR] 梯度出现Inf！")
    elif grad_norm > 1000:
        print(f"[WARNING] 梯度范数过大: {grad_norm}")


def case4_loss_scale_underflow():
    """案例4：Loss Scale持续下降到1"""
    print("\n" + "=" * 60)
    print("案例4：Loss Scale持续下降")
    print("=" * 60)
    
    print("这是一个模拟场景...")
    print("当模型数值不稳定时，GradScaler会不断减半scale")
    print("如果scale降到1还在溢出，说明模型有系统性的数值问题")
    
    scaler = GradScaler()
    scale_history = [65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1, 1]
    
    print(f"\nScale变化: {scale_history}")
    print(f"最终scale: {scale_history[-1]}")
    
    if scale_history[-1] == 1:
        print("[CRITICAL] Scale已归零，训练停滞！")
        print("可能原因：")
        print("1. 学习率过高")
        print("2. 模型初始化问题")
        print("3. 数据预处理问题（数值范围过大）")
        print("4. 模型结构问题（如残差连接未归一化）")


def monitor_nan_inf(tensor, name="tensor"):
    """
    监控tensor的NaN/Inf
    
    Returns:
        (has_nan, has_inf, max_val)
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    max_val = tensor.abs().max().item() if not has_inf else float('inf')
    
    if has_nan:
        print(f"[ALERT] {name} 包含 NaN!")
    if has_inf:
        print(f"[ALERT] {name} 包含 Inf!")
    if not has_nan and not has_inf and max_val > 60000:
        print(f"[WARNING] {name} 接近溢出边界: {max_val}")
    
    return has_nan, has_inf, max_val


if __name__ == "__main__":
    case1_forward_overflow()
    case2_loss_division_by_zero()
    case3_gradient_explosion()
    case4_loss_scale_underflow()
    
    print("\n" + "=" * 60)
    print("检测函数使用示例")
    print("=" * 60)
    
    # 正常tensor
    normal = torch.randn(10, 10)
    print("\n正常tensor检测:")
    monitor_nan_inf(normal, "normal")
    
    # 包含NaN的tensor
    with_nan = torch.tensor([1.0, 2.0, float('nan'), 4.0])
    print("\n包含NaN的tensor检测:")
    monitor_nan_inf(with_nan, "with_nan")
    
    # 接近溢出的tensor
    near_overflow = torch.tensor([60000.0, -60000.0])
    print("\n接近溢出的tensor检测:")
    monitor_nan_inf(near_overflow, "near_overflow")
