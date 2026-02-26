# 训练示例

本目录包含两个完整的训练示例，演示如何在真实模型训练中集成 MixedPrecisionNanny 监控。

## 示例列表

| 示例 | 模型 | 任务 | 说明 |
|------|------|------|------|
| `train_resnet_classification.py` | ResNet-18 | 图像分类 | 合成 CIFAR-10 数据，SGD 优化 |
| `train_yolo_detection.py` | Tiny-YOLO | 目标检测 | 合成检测数据，多分支 Loss |

## 运行方式

```bash
cd MixedPrecisionNanny

# ResNet 分类训练
python examples/train_resnet_classification.py

# YOLO 检测训练
python examples/train_yolo_detection.py
```

## 查询监控结果

训练完成后，可用 CLI 查询监控数据：

```bash
# ResNet 监控结果
python cli.py summary --db examples/resnet_nanny_logs/metrics.db
python cli.py alerts  --db examples/resnet_nanny_logs/metrics.db
python cli.py stats   --db examples/resnet_nanny_logs/metrics.db --step 0

# YOLO 监控结果
python cli.py summary --db examples/yolo_nanny_logs/metrics.db
python cli.py alerts  --db examples/yolo_nanny_logs/metrics.db
```

## 监控集成方式

两个示例都采用 `with nanny.step(step):` 的方式集成监控，只需在训练循环中添加一行：

```python
from nanny import MixedPrecisionNanny

nanny = MixedPrecisionNanny(model, trace_interval=10)

for step, batch in enumerate(dataloader):
    with nanny.step(step):          # ← 唯一侵入点
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

nanny.close()
```

## 合成数据说明

为了保证示例可以在任何环境（包括无 GPU、无网络）下直接运行，两个示例均使用随机合成数据，不依赖外部数据集下载。

- **ResNet 示例**：512 张 3×32×32 随机图像 + 10 类标签
- **YOLO 示例**：256 张 3×128×128 随机图像 + grid-cell 标注（约 20% cell 含目标）
