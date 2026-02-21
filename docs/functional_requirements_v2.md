# TrainingProfiler 功能需求文档 V2

## 修正后的优先级

**P0：静态代码检查**
- 零成本，不依赖GPU显存
- 训练前发现问题（如案例2的loss类型问题）
- 参考PyTorch unsafe ops列表进行扫描

**P1：轻量监控（Decorator方式，自适应阈值）**
- 用户只需`@watch_training`
- 根据训练数据自动确定阈值
- 参考Ling 2.0的上下溢/失真监控

**P2：量化误差分析**
- 针对具体tensor分析维度差异
- 解决案例1的大数吃小数问题

**P3（有条件）：AutoDiff**
- 仅在小模型或部分层可用
- 采用分层对比避免OOM

---

## P0: 静态代码检查

### 检查项（基于PyTorch AMP文档）

| 检查项 | 风险等级 | 检测依据 |
|--------|---------|---------|
| `.half()` / `.bfloat16()` 调用 | 高 | 绕过AMP保护 |
| `binary_cross_entropy` | 高 | AMP中明确禁止 |
| Loss计算在autocast外 | 高 | 类型不一致 |
| `softmax` 无dtype参数 | 中 | FP16下可能溢出 |
| `sum`/`mean` 在autocast外 | 中 | 失去FP32保护 |
| 自定义autograd函数 | 中 | 可能未处理类型 |

### 输出格式
```
File: model.py
Line 45: [HIGH] .half() call detected
  -> 手动类型转换可能绕过AMP保护
  -> 建议：删除.half()，使用with autocast()

Line 67: [HIGH] binary_cross_entropy usage
  -> 该函数在AMP中会报错
  -> 建议：改用binary_cross_entropy_with_logits
```

---

## P1: 轻量监控（自适应阈值装饰器）

### 核心设计：自适应阈值

**问题**：用户不知道应该设置什么阈值（如梯度范数>100是爆炸还是>1000？）

**解决方案**：
1. **预热期**（前N步）：收集基线数据
2. **统计期**：计算均值和标准差
3. **运行时**：基于统计自动确定阈值

### 使用方式

```python
from training_profiler import watch_training

@watch_training(
    warmup_steps=100,      # 前100步收集基线
    alert_window=50,       # 基于最近50步判断异常
)
def train_step(model, batch):
    with autocast():
        loss = model(batch)
    loss.backward()
    optimizer.step()
    return loss
```

**无需配置阈值**，装饰器自动学习正常范围。

### 监控指标（参考Ling 2.0）

#### 1. 上溢检测（Overflow）
**定义**：数值超过FP16/BF16表示范围

```python
# FP16: max = 65504
# BF16: max = 3.4e38 (实际训练很少溢出)

def detect_overflow(tensor):
    if dtype == torch.float16:
        return tensor.abs().max() > 60000  # 接近边界
    return False  # BF16基本不会溢出
```

#### 2. 下溢检测（Underflow）- Ling 2.0核心指标
**定义**：数值小于最小正数，被flush为0

```python
# FP16: min positive = 5.96e-8
# BF16: min positive = 1.17e-38

def detect_underflow(tensor, dtype):
    min_positive = 5.96e-8 if dtype == torch.float16 else 1.17e-38
    underflow_mask = (tensor != 0) & (tensor.abs() < min_positive)
    underflow_ratio = underflow_mask.float().mean()
    return underflow_ratio

# 预警阈值（基于统计自适应）
# 如果某步的下溢比例 > 基线均值 + 3*标准差 → 报警
```

#### 3. 失真检测（Distortion）
**定义**：数值精度损失，相对误差增大

```python
def detect_distortion(current_stats, baseline_stats):
    # 比较当前分布与基线分布的差异
    # 如果均值/方差发生显著变化 → 可能失真
    pass
```

#### 4. 梯度健康度
- 梯度范数趋势（是否偏离基线）
- NaN/Inf检测
- 层级梯度一致性（各层梯度范数比例）

### 自适应阈值算法

```python
class AdaptiveThreshold:
    """基于统计的自适应阈值"""
    
    def __init__(self, warmup_steps=100):
        self.warmup_steps = warmup_steps
        self.step = 0
        self.history = []  # 存储历史指标
        self.baseline = None
    
    def update(self, metric_value):
        """每步调用，更新统计"""
        self.step += 1
        self.history.append(metric_value)
        
        if self.step == self.warmup_steps:
            # 计算基线统计
            self.baseline = {
                'mean': np.mean(self.history),
                'std': np.std(self.history),
                'max': np.max(self.history),
                'min': np.min(self.history),
            }
            print(f"[Monitor] Baseline established: {self.baseline}")
    
    def is_anomaly(self, metric_value, sigma=3):
        """判断是否为异常"""
        if self.baseline is None:
            return False  # 还在预热期
        
        # 基于标准差判断
        deviation = abs(metric_value - self.baseline['mean'])
        if deviation > sigma * self.baseline['std']:
            return True
        
        # 基于范围判断（突然超出历史范围）
        if metric_value > self.baseline['max'] * 2:
            return True
        if metric_value < self.baseline['min'] / 2:
            return True
        
        return False
```

### 输出格式

**正常情况（每100步输出一次）**：
```
[Monitor] Step 300 | grad_norm: 2.34 (baseline: 2.1±0.5) | Status: OK
```

**异常情况（立即输出）**：
```
[ALERT] Step 305 | grad_norm: 15.6 (baseline: 2.1±0.5) | Deviation: 27σ
[ALERT] Step 305 | Layer 'encoder.3' underflow_ratio: 0.05 (baseline: 0.001±0.0005)
```

### 性能保证

- Hook开销：< 1ms/步
- 内存开销：只存统计值，不存完整tensor
- CPU开销：异步统计，不影响训练

---

## P2: 量化误差分析

针对案例1的维度不平衡问题。

### 功能

```python
from training_profiler import analyze_tensor

# 分析特定tensor
report = analyze_tensor(tensor, name="bbox_pred")

# 输出示例
report = {
    'shape': (1000, 2),
    'dim_stats': {
        0: {'min': -1000, 'max': 1200, 'mean': 100},
        1: {'min': -1.5, 'max': 0.8, 'mean': 0.1},
    },
    'dim_ratio': 1500,  # 维度间最大比值
    'risk_level': 'HIGH',
    'suggestion': '维度1数值范围小，reduce时建议用FP32',
}
```

---

## 文件结构

```
TrainingProfiler/
├── skills/                          # 固化的知识和案例
│   ├── unsafe_ops/                  # PyTorch unsafe ops列表
│   │   └── README.md
│   ├── numerical_issues/            # 数值问题分类和案例
│   │   ├── README.md
│   │   ├── case_dim_imbalance.py   # 维度不平衡案例
│   │   └── case_nan_inf.py         # NaN/Inf案例
│   ├── reduction_patterns/          # Reduce操作问题模式
│   │   └── README.md
│   └── common_fixes/                # 常见修复方案
│       └── README.md
├── static_checker.py                # P0: 静态代码检查
├── adaptive_monitor.py              # P1: 自适应监控装饰器
├── tensor_analyzer.py               # P2: Tensor分析工具
└── docs/
    └── functional_requirements_v2.md # 本文档
```

---

## 与Ling 2.0的对比

| 特性 | Ling 2.0 (生产框架) | TrainingProfiler (诊断工具) |
|------|-------------------|---------------------------|
| 目标 | FP8训练的生产保障 | BF16/FP16问题诊断 |
| 精度监控 | tile/block wise | tensor维度级 |
| 阈值设定 | 预定义+人工调参 | 自适应学习 |
| 使用方式 | 内嵌框架 | 装饰器/脚本 |
| 侵入性 | 深度集成 | 零侵入（装饰器） |

**核心差异**：Ling 2.0需要修改训练框架，我们是外挂式诊断工具。
