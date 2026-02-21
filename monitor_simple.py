"""
极简梯度监控
用法：
    from monitor_simple import watch_model
    watch_model(model)
    # 然后正常训练
"""
import torch
import sys

# 全局状态
_step = 0
_has_alert = False


def watch_model(model, alert_threshold=100.0, vanish_threshold=1e-6):
    """
    给模型注册监控hook
    
    Args:
        model: PyTorch模型
        alert_threshold: 梯度范数超过此值报警（梯度爆炸）
        vanish_threshold: 梯度范数低于此值报警（梯度消失）
    """
    
    def make_hook(name):
        def hook(grad):
            global _step, _has_alert
            _step += 1
            
            # 检查NaN/Inf
            if torch.isnan(grad).any():
                print(f"\n[ALERT] {name}: NaN detected!", file=sys.stderr)
                _has_alert = True
            if torch.isinf(grad).any():
                print(f"\n[ALERT] {name}: Inf detected!", file=sys.stderr)
                _has_alert = True
            
            # 检查梯度范数
            norm = grad.norm().item()
            if norm > alert_threshold:
                print(f"\n[ALERT] {name}: Gradient explosion (norm={norm:.2f})", file=sys.stderr)
                _has_alert = True
            elif norm < vanish_threshold and norm > 0:
                print(f"\n[ALERT] {name}: Gradient vanishing (norm={norm:.2e})", file=sys.stderr)
                _has_alert = True
            
            # 每1000步打印一次正常信息
            if _step % 1000 == 0 and not _has_alert:
                print(f"[INFO] step {_step}: {name} grad_norm={norm:.4f}")
            
            return grad
        return hook
    
    # 给所有参数注册hook
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(make_hook(name))
            count += 1
    
    print(f"[INFO] Monitoring {count} parameters")


def check_loss_scale(scaler):
    """
    检查GradScaler状态
    
    Args:
        scaler: torch.cuda.amp.GradScaler实例
    """
    scale = scaler.get_scale()
    
    if scale < 1.0:
        print(f"\n[WARNING] Loss scale dropped to {scale}. Training may be unstable.", file=sys.stderr)
        return False
    elif scale < 128:
        print(f"\n[WARNING] Loss scale is low ({scale}). Watch for underflow.", file=sys.stderr)
        return False
    else:
        print(f"[INFO] Loss scale: {scale}")
        return True


def check_forward_output(module, input, output):
    """
    检查前向输出（注册为forward hook使用）
    
    用法:
        model.register_forward_hook(check_forward_output)
    """
    global _step
    
    if isinstance(output, torch.Tensor):
        # 检查是否有Inf/NaN
        if torch.isnan(output).any():
            print(f"\n[ALERT] {module.__class__.__name__} output has NaN!", file=sys.stderr)
        if torch.isinf(output).any():
            print(f"\n[ALERT] {module.__class__.__name__} output has Inf!", file=sys.stderr)
        
        # 检查数值范围（接近FP16边界）
        max_val = output.abs().max().item()
        if max_val > 60000:  # FP16 max is 65504
            print(f"\n[WARNING] {module.__class__.__name__} output near FP16 max: {max_val}", file=sys.stderr)


def save_debug_data(filepath="debug_grads.pt"):
    """
    保存当前梯度状态用于事后分析
    """
    debug_info = {
        'step': _step,
        'has_alert': _has_alert,
    }
    torch.save(debug_info, filepath)
    print(f"[INFO] Debug data saved to {filepath}")
