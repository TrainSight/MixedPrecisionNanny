# Reduce操作的精度问题

Reduce操作（sum/mean等）是混合精度训练中的高风险点。

## 问题本质

Reduce操作将多个数合并成一个数，在FP16/BF16中容易出现：
1. **大数吃小数**：小数值在累加过程中被忽略
2. **累积误差**：多次累加导致精度损失
3. **溢出**：累加结果超过表示范围

## 高风险Reduce模式

### 模式1：跨维度求和（维度不平衡场景）

```python
# 高风险代码
tensor = torch.randn(1000, 2)  # 维度0数量大，维度1数量小
# 假设维度0数值范围大，维度1数值范围小
result = tensor.sum(dim=0)  # 维度1可能被影响
```

**风险**：如果某些维度数值范围差异大，小数值维度精度受损

### 模式2：序列长Reduce

```python
# 长序列的累积
seq = torch.randn(1, 10000)  # 长序列
result = seq.sum(dim=-1)  # 累加10000次，误差累积
```

**风险**：累加次数越多，误差累积越大

### 模式3：混合正负数的Reduce

```python
# 正负抵消场景
x = torch.tensor([1000.0, -999.9, 0.1])
result = x.sum()  # 理论结果0.2，实际可能有误差
```

**风险**：大数正负抵消时，小数精度受影响

## PyTorch的默认保护

PyTorch AMP会自动将以下操作转为FP32：
```python
sum, mean, cumsum, cumprod, prod
```

**但是**，如果用户：
1. 在autocast外执行reduce
2. 手动转换了类型
3. 自定义了reduce逻辑

就会失去这个保护。

## 检测建议

对于reduce操作，建议检查：
1. 输入tensor的数值范围
2. reduce的维度大小
3. 输入中是否存在离群值（outlier）

**检测代码**：
```python
def check_reduce_safety(tensor, dim):
    """检查reduce操作的安全性"""
    warnings = []
    
    # 检查数值范围
    max_val = tensor.abs().max()
    min_val_nonzero = tensor[tensor != 0].abs().min()
    
    if max_val / min_val_nonzero > 1e4:
        warnings.append("数值范围差异大，建议用FP32")
    
    # 检查reduce维度大小
    dim_size = tensor.shape[dim]
    if dim_size > 10000:
        warnings.append(f"reduce维度大({dim_size})，累积误差风险")
    
    # 检查离群值
    mean = tensor.mean()
    std = tensor.std()
    outliers = (tensor.abs() > mean + 3 * std).sum()
    if outliers > 0:
        warnings.append(f"存在{outliers}个离群值，可能影响精度")
    
    return warnings
```

## 修复方案

### 方案1：使用FP32执行reduce

```python
# 在reduce前转FP32
result = tensor.float().sum(dim=-1)
```

### 方案2：分块reduce（Kahan求和）

对于特别大的reduce，可以分块进行：

```python
def safe_sum_large(tensor, dim=-1, chunk_size=1000):
    """分块求和，减少累积误差"""
    chunks = torch.split(tensor, chunk_size, dim=dim)
    partial_sums = [chunk.sum(dim=dim) for chunk in chunks]
    return torch.stack(partial_sums).sum(dim=0)
```

### 方案3：避免在关键路径reduce

如果可能，修改模型结构避免大范围的reduce操作。
