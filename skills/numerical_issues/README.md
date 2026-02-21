# 混合精度数值问题分类

## 问题类型总览

```
混合精度训练问题
├── 显性错误（训崩）
│   ├── NaN/Inf 传播
│   ├── 梯度爆炸
│   └── Loss Scale 归零
├── 隐性错误（训慢/效果差）
│   ├── 梯度消失
│   ├── 大数吃小数
│   ├── 维度不平衡
│   └── Loss Scale 持续下降
└── 类型不匹配
    ├── 手动类型转换
    └── Loss计算类型不一致
```

## 1. 显性错误

### 1.1 NaN/Inf 传播

**现象**：Loss 变成 NaN，或某层输出为 Inf

**根因**：
- 前向传播中数值溢出（>65504 for FP16）
- 梯度计算中出现除零
- Loss Scale 调整后仍溢出

**检测方法**：
```python
# 检查NaN/Inf
has_nan = torch.isnan(tensor).any()
has_inf = torch.isinf(tensor).any()

# 检查数值边界
near_overflow = (tensor.abs() > 60000).any()  # FP16
```

**案例代码**：见 `case_nan_inf.py`

### 1.2 梯度爆炸

**现象**：梯度范数突然增大，参数更新失控

**判断标准**：
```
梯度范数 > 100：危险
梯度范数 > 1000：即将崩溃
```

**检测方法**：
```python
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.norm().item() ** 2
total_norm = total_norm ** 0.5

if total_norm > 100:
    alert("Gradient explosion detected")
```

### 1.3 Loss Scale 归零（FP16）

**现象**：GradScaler 的 scale 值降到 1.0 并维持

**根因**：模型中存在系统性溢出，无法通过降低 scale 解决

**判断标准**：
```
scale = 1.0 且持续多步 → 严重，训练已停滞
scale < 128 且持续下降 → 警告，数值不稳定
```

**检测方法**：
```python
if scaler.get_scale() < 1.0:
    alert("Loss scale underflow, training stalled")
```

## 2. 隐性错误

### 2.1 梯度消失

**现象**：Loss 不下降或下降极慢，模型"学不动"

**判断标准**：
```
梯度范数 < 1e-6：严重消失
梯度范数 < 0.01：轻微消失
```

**特殊检测**：
```python
# 检查某些层是否完全无梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        if param.grad.abs().max() < 1e-10:
            alert(f"Layer {name} has no effective gradient")
```

### 2.2 大数吃小数（Catastrophic Cancellation）

**现象**：数值精度丢失，信息被"吞掉"

**经典场景**：
```python
# 问题代码
large = torch.tensor([10000.0])
small = torch.tensor([0.1])
result = large + small  # FP16: 10000.0，small被吃掉
```

**检测方法**：
```python
# 检查数值范围差异
max_val = tensor.abs().max()
min_val_nonzero = tensor[tensor != 0].abs().min()
ratio = max_val / min_val_nonzero

if ratio > 1e4:
    alert(f"Large ratio {ratio}, risk of precision loss")
```

### 2.3 维度不平衡

**现象**：Tensor 不同维度间数值范围差异大

**案例**：
- 维度0（x轴）：[-1000, 1000]
- 维度1（y轴）：[-1, 1]

**根因**：物理意义导致的数值差异，在 reduce 时小数被吃

**检测方法**：
```python
# 按维度检查数值范围
for dim in range(tensor.ndim):
    dim_max = tensor.abs().max(dim=dim)[0]
    dim_min = tensor[tensor != 0].abs().min(dim=dim)[0]
    ratio = dim_max / dim_min
    
    if ratio.max() > 1e4:
        alert(f"Dimension {dim} has unbalanced range")
```

**案例代码**：见 `case_dim_imbalance.py`

### 2.4 Loss Scale 持续下降

**现象**：scale 值不断减半，虽未到1但已很低

**判断标准**：
```
连续5次调整都是下降 → 警告
scale < 初始值/100 → 严重
```

**含义**：模型数值不稳定，频繁溢出

## 3. 类型不匹配

### 3.1 手动类型转换

**现象**：用户显式调用 `.half()` 或 `.bfloat16()`

**风险**：
- 绕过AMP的自动类型提升
- 可能与其他FP32部分产生类型冲突
- 梯度缩放失效

**检测**：AST扫描 `.half()` 调用

### 3.2 Loss计算类型不一致

**现象**：
```python
with autocast():
    output = model(input)  # fp16
loss = loss_fn(output, target)  # output fp16, target fp32
```

**风险**：某些loss函数对类型敏感，可能产生异常

**检测**：检查loss计算是否在autocast上下文内

## 4. 问题严重性分级

| 问题 | 级别 | 影响 | 检测难度 |
|------|------|------|---------|
| NaN/Inf | P0 | 训练崩溃 | 容易 |
| Loss Scale归零 | P0 | 训练停滞 | 容易 |
| 梯度爆炸 | P0 | 模型崩溃 | 容易 |
| 梯度消失 | P1 | 学不动 | 中等 |
| 大数吃小数 | P1 | 效果差 | 困难 |
| 维度不平衡 | P1 | 特定层失效 | 困难 |
| 类型不匹配 | P1 | 潜在风险 | 容易（静态） |

## 5. 监控建议

**P0（必须监控）**：
- 每步检查 NaN/Inf
- 每步检查 Loss Scale 状态
- 每步检查梯度范数

**P1（建议监控）**：
- 每100步检查梯度分布
- 每100步检查激活值范围
- 定期分析tensor维度差异

**P2（问题排查时用）**：
- FP32 vs BF16 逐层对比
- 特定tensor的精度分析
