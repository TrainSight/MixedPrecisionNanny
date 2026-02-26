"""
简化 YOLO 目标检测训练示例 + MixedPrecisionNanny 监控

功能演示：
  1. 简化版 YOLO 检测模型在合成目标检测数据上训练
  2. 集成 MixedPrecisionNanny 实时监控梯度、激活值
  3. 展示检测模型特有的多分支 loss（bbox + objectness + classification）

模型结构：
  - Backbone: 轻量 CNN 提取特征
  - Detection Head: 1×1 Conv 输出 (5 + num_classes) 通道
    - 5 = [tx, ty, tw, th, objectness]
    - num_classes 个类别概率

运行：
    cd MixedPrecisionNanny
    python examples/train_yolo_detection.py

查询结果：
    python cli.py summary --db examples/yolo_nanny_logs/metrics.db
    python cli.py alerts  --db examples/yolo_nanny_logs/metrics.db
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from nanny import MixedPrecisionNanny


# ─── YOLO-style Model ───────────────────────────────────────────────────────


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TinyYOLOBackbone(nn.Module):
    """轻量 Darknet 风格 backbone，将 128×128 → 8×8 feature map。"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBnRelu(3, 16, 3, stride=2),     # 128→64
            ConvBnRelu(16, 32, 3, stride=2),    # 64→32
            ConvBnRelu(32, 64, 3, stride=2),    # 32→16
            ConvBnRelu(64, 128, 3, stride=2),   # 16→8
            ConvBnRelu(128, 256, 3, stride=1),  # 8→8
        )

    def forward(self, x):
        return self.layers(x)


class YOLOHead(nn.Module):
    """
    YOLO 检测头：每个 grid cell 预测 num_boxes 个 bbox。
    输出: [B, num_boxes * (5 + num_classes), H, W]
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 5, num_boxes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        out_channels = num_boxes * (5 + num_classes)
        self.conv = nn.Sequential(
            ConvBnRelu(in_channels, in_channels, 3),
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.conv(x)


class TinyYOLO(nn.Module):
    """简化版 YOLO 目标检测模型。"""

    def __init__(self, num_classes: int = 5, num_boxes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.backbone = TinyYOLOBackbone()
        self.head = YOLOHead(
            in_channels=256, num_classes=num_classes, num_boxes=num_boxes
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ─── YOLO Loss ───────────────────────────────────────────────────────────────


class YOLOLoss(nn.Module):
    """
    简化版 YOLO Loss：
      - bbox_loss:  MSE on (tx, ty, tw, th)
      - obj_loss:   BCE on objectness
      - cls_loss:   CrossEntropy on class predictions
    """

    def __init__(self, num_classes: int = 5, num_boxes: int = 2, lambda_coord: float = 5.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.per_box = 5 + num_classes
        self.lambda_coord = lambda_coord
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, pred, target_boxes, target_obj, target_cls):
        """
        Args:
            pred:         [B, num_boxes*(5+C), H, W]
            target_boxes: [B, num_boxes, 4, H, W]  — (tx, ty, tw, th)
            target_obj:   [B, num_boxes, H, W]      — objectness (0 or 1)
            target_cls:   [B, num_boxes, H, W]      — class index (long)
        """
        B, _, H, W = pred.shape
        pred = pred.view(B, self.num_boxes, self.per_box, H, W)

        pred_box = pred[:, :, :4, :, :]
        pred_obj = pred[:, :, 4, :, :]
        pred_cls = pred[:, :, 5:, :, :]

        bbox_loss = self.mse(pred_box, target_boxes)
        obj_loss = self.bce(pred_obj, target_obj)

        pred_cls_flat = pred_cls.permute(0, 1, 3, 4, 2).reshape(-1, self.num_classes)
        target_cls_flat = target_cls.reshape(-1)
        cls_loss = self.ce(pred_cls_flat, target_cls_flat)

        total = self.lambda_coord * bbox_loss + obj_loss + cls_loss
        return total, bbox_loss.item(), obj_loss.item(), cls_loss.item()


# ─── Synthetic Detection Dataset ─────────────────────────────────────────────


class SyntheticDetectionDataset(Dataset):
    """
    合成目标检测数据：
      - 128×128 RGB 图像（随机噪声）
      - 每个 grid cell 随机标注 0~1 个目标
      - 约 20% 的 cell 包含目标
    """

    def __init__(
        self,
        num_samples: int = 256,
        grid_size: int = 8,
        num_classes: int = 5,
        num_boxes: int = 2,
        obj_ratio: float = 0.2,
    ):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.obj_ratio = obj_ratio

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, 128, 128)
        G, nb, nc = self.grid_size, self.num_boxes, self.num_classes

        target_boxes = torch.zeros(nb, 4, G, G)
        target_obj = torch.zeros(nb, G, G)
        target_cls = torch.zeros(nb, G, G, dtype=torch.long)

        obj_mask = torch.rand(nb, G, G) < self.obj_ratio
        target_obj[obj_mask] = 1.0
        target_boxes[:, 0][obj_mask] = torch.rand(obj_mask.sum())  # tx
        target_boxes[:, 1][obj_mask] = torch.rand(obj_mask.sum())  # ty
        target_boxes[:, 2][obj_mask] = torch.rand(obj_mask.sum()) * 0.5  # tw
        target_boxes[:, 3][obj_mask] = torch.rand(obj_mask.sum()) * 0.5  # th
        target_cls[obj_mask] = torch.randint(0, nc, (obj_mask.sum(),))

        return image, target_boxes, target_obj, target_cls


# ─── Training Loop ───────────────────────────────────────────────────────────


def train(
    num_epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-3,
    trace_interval: int = 10,
    output_dir: str = "examples/yolo_nanny_logs",
):
    torch.manual_seed(42)
    device = torch.device("cpu")

    print("=" * 60)
    print("  Tiny-YOLO Detection Training with Nanny Monitoring")
    print("=" * 60)

    num_classes = 5
    num_boxes = 2
    model = TinyYOLO(num_classes=num_classes, num_boxes=num_boxes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    dataset = SyntheticDetectionDataset(
        num_samples=256,
        num_classes=num_classes,
        num_boxes=num_boxes,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = YOLOLoss(num_classes=num_classes, num_boxes=num_boxes)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    nanny = MixedPrecisionNanny(
        model,
        trace_interval=trace_interval,
        output_dir=output_dir,
        verbose=True,
    )

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_bbox = 0.0
        epoch_obj = 0.0
        epoch_cls = 0.0

        for batch_idx, (images, t_boxes, t_obj, t_cls) in enumerate(dataloader):
            images = images.to(device)
            t_boxes = t_boxes.to(device)
            t_obj = t_obj.to(device)
            t_cls = t_cls.to(device)

            with nanny.step(global_step):
                optimizer.zero_grad()
                pred = model(images)
                loss, bbox_l, obj_l, cls_l = criterion(pred, t_boxes, t_obj, t_cls)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            epoch_bbox += bbox_l
            epoch_obj += obj_l
            epoch_cls += cls_l
            global_step += 1

        n = len(dataloader)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}]  "
            f"Loss: {epoch_loss / n:.4f}  "
            f"(bbox: {epoch_bbox / n:.4f}  "
            f"obj: {epoch_obj / n:.4f}  "
            f"cls: {epoch_cls / n:.4f})"
        )

    nanny.close()

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Monitoring data saved to: {output_dir}/metrics.db")
    print("  Query results with:")
    print(f"    python cli.py summary --db {output_dir}/metrics.db")
    print(f"    python cli.py alerts  --db {output_dir}/metrics.db")
    print("=" * 60)


if __name__ == "__main__":
    train()
