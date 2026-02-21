# TrainingProfiler 功能需求文档

## 背景

基于实际业务中混合精度训练（FP16/BF16）的排查痛点，开发一套轻量级诊断工具。

## 核心功能

### 功能1：自动逐层对比（Auto Diff）

**解决的问题**：案例1中需要人工写hook dump、人工diff比对的问题

**功能描述**：
自动对比FP32和BF16/FP16前向输出，定位具体哪一层、哪个tensor、哪个维度出现数值差异。

**输入**：
- 模型实例
- 样本输入数据
- 对比精度（BF16或FP16）

**输出**：
- 每层输出的相对误差统计
- 标记异常tensor（误差>阈值）
- 异常tensor的维度分析（max/min per dimension）

**关键阈值**：
- 相对误差 > 1%：警告
- 相对误差 > 10%：严重
- 某维度max/min比值 > 1e4：可能存在大数吃小数

**实现要点**：
```python
# 伪代码
def auto_diff(model, sample_input, target_dtype=torch.bfloat16):
    # FP32基准
    output_fp32 = model(sample_input)
    
    # 低精度输出
    with torch.autocast(dtype=target_dtype):
        output_amp = model(sample_input)
    
    # 逐层hook收集中间结果并对比
    # 返回差异报告
```

---

### 功能2：静态代码检查（Static Check）

**解决的问题**：案例2中loss代码未转换到BF16的问题

**功能描述**：
扫描模型代码，标记混合精度训练中的常见风险点。

**检查项**：

| 检查项 | 风险等级 | 说明 |
|--------|---------|------|
| 手动.half()调用 | 高 | 可能跳过AMP保护机制 |
| loss计算未在autocast内 | 高 | 类型不一致导致NaN |
| softmax/log_softmax无dtype | 中 | FP16下可能溢出 |
| binary_cross_entropy | 高 | AMP中不稳定 |
| LayerNorm/BatchNorm后无检查 | 低 | 需关注输入范围 |
| Reduce操作（sum/mean） | 中 | 大数吃小数风险 |

**输出**：
- 风险点位置（文件、行号）
- 风险说明
- 修复建议

---

### 功能3：训练时监控（Runtime Monitor）

**解决的问题**：训练中隐性数值问题（如loss scale持续下降、某些层梯度消失）

**功能描述**：
轻量级hook，实时监控训练过程中的关键指标。

**监控指标**：

#### 3.1 梯度健康度
- 各层梯度范数（L2 norm）
- 全局梯度范数
- NaN/Inf检测

**判断标准**：
```
梯度范数 > 100：爆炸警告
梯度范数 < 1e-6：消失警告
某层梯度为0但其他层正常：该层死亡警告
```

#### 3.2 Loss Scale监控（FP16）
- 当前scale值
- 调整历史（上升/下降次数）

**判断标准**：
```
scale < 128：警告（频繁溢出）
scale = 1：严重（无法正常训练）
连续下降超过5次：模型数值不稳定
```

#### 3.3 激活值范围（可选，性能敏感时关闭）
- 每层输出的min/max
- 接近FP16/BF16边界（>60000）警告

**输出**：
- 实时日志（stderr输出，无需前端）
- 异常时保存debug数据（可选）

---

### 功能4：量化误差分析（参考Ling 2.0 FP8思路，适配BF16/FP16）

**解决的问题**：案例1中维度相关的大数吃小数问题

**功能描述**：
分析tensor在高低精度转换时的量化误差分布。

**分析维度**：

#### 4.1 下溢检测（Underflow）
- 检测有多少数值小于目标精度的最小表示值
- BF16最小值：~1.17e-38
- FP16最小值：~5.96e-8

**判断标准**：
```
下溢比例 > 1%：警告
下溢比例 > 10%：严重
```

#### 4.2 维度差异分析
- 按维度统计max/min
- 检测维度间数值差异（如x轴1000 vs y轴1）

**判断标准**：
```
维度间max比值 > 1e4：可能存在reduce时的大数吃小数
```

#### 4.3 Reduce操作精度分析
- 标记sum/mean等操作
- 建议在数值范围大时使用FP32

**输出**：
- 每个风险tensor的分析报告
- 具体的维度差异数据
- 优化建议（如"建议在dim=1上做sum时先转FP32"）

---

## 非功能性需求

### 1. 简洁性原则
- 无复杂封装，纯函数式
- 每个功能独立脚本，可单独运行
- 总代码量控制在1000行以内

### 2. 性能要求
- 监控功能对训练速度影响 < 5%
- 自动对比功能离线运行，不干扰训练

### 3. 依赖最小化
- 仅依赖PyTorch
- 可视化使用matplotlib（无需web前端）

---

## 功能优先级

### P0（核心，先实现）
1. **自动逐层对比**：解决目前最大的排查痛点
2. **基础训练监控**：梯度NaN检测、Loss Scale监控

### P1（重要，后实现）
3. **静态代码检查**：提前发现问题
4. **量化误差分析**：解决维度相关的大数吃小数

### P2（可选）
5. 简单的可视化脚本（matplotlib画loss曲线、梯度范数趋势）

---

## 使用场景示例

### 场景1：训练前检查
```bash
# 检查模型代码
python check_model.py model.py

# 自动对比FP32 vs BF16
python auto_diff.py --model model.py --input sample.pt --dtype bf16
```

### 场景2：训练中监控
```python
from training_profiler import watch_model, check_loss_scale

model = MyModel()
watch_model(model)  # 注册监控hook

scaler = GradScaler()
for batch in dataloader:
    with autocast():
        loss = model(batch)
    
    scaler.scale(loss).backward()
    check_loss_scale(scaler)  # 检查scale状态
    scaler.step(optimizer)
    scaler.update()
```

### 场景3：排查具体tensor问题
```python
from training_profiler import analyze_tensor

# 分析某个tensor的量化误差
analyze_tensor(tensor, target_dtype=torch.bfloat16)
# 输出：各维度max/min、下溢比例、失真比例
```

---

## 参考：Ling 2.0 FP8的启示

Ling 2.0在FP8训练中采用的监控思路（适配到BF16/FP16）：

| Ling 2.0做法 | 我们的适配 |
|-------------|-----------|
| 细粒度tile/block wise scaling监控 | 维度级别的数值范围分析 |
| 下溢/失真监控 | 检测数值是否接近精度边界 |
| 高精度重算对比 | FP32 vs BF16自动对比 |
| loss diff跟踪 | 训练中实时loss监控 |

---

## 边界说明

**不支持的功能**（明确排除）：
- 分布式训练专项诊断
- FP8支持（硬件限制）
- Web前端可视化
- 自动修复代码

**定位**：开发者工具，辅助人工排查，不替代人工判断。
