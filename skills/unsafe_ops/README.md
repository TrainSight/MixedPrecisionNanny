# PyTorch AMP 算子精度分类

参考 PyTorch 官方文档：https://docs.pytorch.org/docs/stable/amp.html

## 1. 强制转 FP32 的算子（PyTorch已保护）

这些算子在 `autocast` 区域内会被 PyTorch **自动**转换为 FP32 执行：

### 归约操作（Reduction）
```python
sum, mean, cumsum, cumprod, prod
cross_entropy, nll_loss, l1_loss, mse_loss
softmax, log_softmax
```

### 数值敏感操作
```python
exp, log, log1p, log2, log10
pow, sqrt, rsqrt, reciprocal
sin, cos, tan, sinh, cosh
acos, asin
```

### 归一化层
```python
layer_norm, batch_norm, group_norm, instance_norm
```

### 距离/相似度计算
```python
cdist, pdist, cosine_similarity, cosine_embedding_loss
```

## 2. 在 FP16/BF16 中执行的算子（安全且快速）

```python
linear, conv1d, conv2d, conv3d
matmul, mm, bmm, mv
relu, gelu, silu
```

## 3. 类型提升算子（多输入时以最宽类型为准）

```python
cat, stack, addcmul, addcdiv
bilinear, cross, dot, tensordot
```

**风险点**：如果混合了 FP32 和 FP16 输入，输出会是 FP32，可能引入类型不匹配。

## 4. 明确禁止或不安全的算子

### 严格禁止（会报错）
```python
binary_cross_entropy  # 必须使用 binary_cross_entropy_with_logits
```

### 需要特别注意
```python
# 这些操作在 FP16 下容易溢出，虽然PyTorch会处理，但需要注意输入范围
softmax  # 如果输入值太大（>10000），即使FP32也会出问题
```

## 5. 用户代码中常见的unsafe模式

### 5.1 手动类型转换（危险）
```python
# ❌ 错误：绕过AMP保护
x = x.half()  # 或 .cuda().half()

# ✅ 正确：让AMP自动处理
with autocast():
    output = model(input)
```

### 5.2 Loss计算在autocast外（危险）
```python
# ❌ 错误：类型不一致
with autocast():
    output = model(input)
loss = loss_fn(output, target)  # output是fp16，target是fp32

# ✅ 正确：loss也在autocast内
with autocast():
    output = model(input)
    loss = loss_fn(output, target)
```

### 5.3 自定义autograd函数未处理类型（危险）
```python
# ❌ 错误：未考虑类型
class MyFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2

# ✅ 正确：使用autocast上下文
class MyFunction(Function):
    @staticmethod
    @torch.autocast(device_type='cuda')  # 或手动处理类型
    def forward(ctx, x):
        return x * 2
```

## 6. 检查清单

在代码扫描时，需要标记：
- [ ] `.half()` 或 `.bfloat16()` 调用
- [ ] `binary_cross_entropy` 使用
- [ ] autocast 上下文外的 loss 计算
- [ ] 自定义 CUDA 算子（未标记为 AMP 安全）
- [ ] Reduce 操作后未做数值检查
