# 混合精度问题常见修复方案

## 修复方案速查表

| 问题 | 快速修复 | 详细说明 |
|------|---------|---------|
| 手动.half() | 删除，改用autocast | 见下方"移除手动类型转换" |
| Loss类型不匹配 | 确保loss在autocast内 | 见"修复Loss计算" |
| 梯度爆炸 | 添加梯度裁剪 | `clip_grad_norm_(model.parameters(), max_norm=1.0)` |
| Loss Scale归零 | 检查学习率或初始化 | 见"诊断Loss Scale问题" |
| 大数吃小数 | reduce前转FP32 | `tensor.float().sum()` |
| 维度不平衡 | 分块处理或调整结构 | 见"处理维度不平衡" |

## 1. 移除手动类型转换

### 问题代码
```python
model = model.half()  # 或 .cuda().half()
input = input.half()
```

### 修复代码
```python
# 删除.half()调用
model = model.cuda()

# 使用autocast自动处理
with autocast():
    output = model(input)
```

### 为什么
手动`.half()`会：
- 绕过AMP的自动类型提升
- 导致与其他FP32组件的类型冲突
- 使GradScaler失效

## 2. 修复Loss计算

### 问题代码
```python
with autocast():
    output = model(input)
loss = criterion(output, target)  # 在autocast外
```

### 修复代码
```python
with autocast():
    output = model(input)
    loss = criterion(output, target)  # 在autocast内
```

### 特别注意
某些loss函数在AMP中有特殊处理：
```python
# binary_cross_entropy 在AMP中会报错
# 改用 binary_cross_entropy_with_logits

# 问题代码
loss = F.binary_cross_entropy(pred, target)

# 修复代码
loss = F.binary_cross_entropy_with_logits(logits, target)
```

## 3. 添加梯度裁剪

### 问题
梯度范数过大（>100），可能导致训练不稳定

### 修复代码
```python
from torch.nn.utils import clip_grad_norm_

# 在optimizer.step()前添加
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # 必须先unscale
clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### 选择max_norm
- 一般模型：1.0
- 大模型/不稳定模型：0.5
- 小模型/稳定模型：5.0

## 4. 诊断Loss Scale问题

### 现象
GradScaler的scale值持续下降，最终到1

### 诊断步骤
1. 检查学习率是否过高
   ```python
   # 尝试降低学习率到原来的1/10
   optimizer = AdamW(model.parameters(), lr=1e-5)  # 原为1e-4
   ```

2. 检查数据预处理
   ```python
   # 确保输入数据归一化
   input = (input - input.mean()) / input.std()
   ```

3. 检查模型初始化
   ```python
   # 使用适当的初始化
   def init_weights(m):
       if isinstance(m, nn.Linear):
           nn.init.xavier_uniform_(m.weight)
           nn.init.zeros_(m.bias)
   
   model.apply(init_weights)
   ```

4. 检查残差连接
   ```python
   # 确保残差连接有适当的归一化
   class ResidualBlock(nn.Module):
       def forward(self, x):
           return x + self.layer(self.norm(x))  # 有LayerNorm
   ```

## 5. 处理Reduce精度问题

### 问题代码
```python
# 数值范围差异大的tensor
result = tensor.sum(dim=-1)  # 可能丢失精度
```

### 修复代码
```python
# 方案1：转FP32后再reduce
result = tensor.float().sum(dim=-1)

# 方案2：使用更稳定的累加方式（如果tensor很大）
def stable_sum(tensor, dim=-1):
    # 分块求和减少累积误差
    chunk_size = 1000
    chunks = torch.split(tensor, chunk_size, dim=dim)
    return torch.stack([c.sum(dim=dim) for c in chunks]).sum(dim=0)
```

## 6. 处理维度不平衡

### 问题代码
```python
# x轴范围大，y轴范围小
data = torch.randn(1000, 2)  # [batch, 2] 2表示(x,y)
loss = data.sum(dim=0)  # y轴精度可能受损
```

### 修复代码
```python
# 方案1：分别处理不同维度
x_loss = data[:, 0].float().sum()  # x轴用FP32
y_loss = data[:, 1].float().sum()  # y轴用FP32

# 方案2：标准化后再处理
mean = data.mean(dim=0, keepdim=True)
std = data.std(dim=0, keepdim=True)
normalized = (data - mean) / (std + 1e-8)
loss = normalized.sum(dim=0)
```

## 7. 通用检查清单

在排查混合精度问题时，按顺序检查：

1. [ ] 移除了所有`.half()`/`.bfloat16()`调用
2. [ ] Loss计算在autocast内
3. [ ] 使用了`binary_cross_entropy_with_logits`而非`binary_cross_entropy`
4. [ ] 添加了梯度裁剪（如果模型不稳定）
5. [ ] 检查Loss Scale变化趋势
6. [ ] 输入数据已归一化
7. [ ] 模型初始化合适
8. [ ] Reduce操作前数值范围检查
9. [ ] 没有维度不平衡问题
