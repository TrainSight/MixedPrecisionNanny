# TrainingProfiler MCP Server

通过MCP协议提供混合精度训练的静态代码检查能力。

## 架构说明

```
┌─────────────────┐     MCP Protocol      ┌──────────────────┐
│   AI Agent      │  ◄─────────────────►  │  MCP Server      │
│   (Claude/Other)│                       │  (this folder)   │
└─────────────────┘                       └──────────────────┘
                                                  │
                         ┌────────────────────────┼────────────────────────┐
                         ▼                        ▼                        ▼
                  ┌─────────────┐        ┌──────────────┐        ┌──────────────┐
                  │   Tools     │        │  Resources   │        │   Prompts    │
                  │ - analyze   │        │ - unsafe ops │        │ - static     │
                  │   _code     │        │   list       │        │   check      │
                  │ - search_   │        │ - case       │        │   guide      │
                  │   pattern   │        │   studies    │        │              │
                  │ - get_fix_  │        │ - fix        │        │              │
                  │ suggestion  │        │   recipes    │        │              │
                  └─────────────┘        └──────────────┘        └──────────────┘
```

## 提供的功能

### Tools（工具）

Agent可以调用的工具：

1. **`analyze_code`** - 综合分析代码
   - 输入：Python代码
   - 输出：风险报告（问题列表、严重级别、行号）

2. **`search_pattern`** - 搜索特定风险模式
   - `half_call`: 手动.half()调用
   - `bce_loss`: 不安全的binary_cross_entropy
   - `autocast_scope`: autocast作用域分析
   - `reduce_op`: reduce操作

3. **`get_fix_suggestion`** - 获取修复建议
   - 输入：问题类型
   - 输出：修复代码示例

4. **`check_file`** - 读取并分析文件
   - 输入：文件路径
   - 输出：文件内容和元数据

### Resources（知识库）

Agent可以读取的知识资源：

- `skills://unsafe_ops/pytorch_amp` - PyTorch AMP算子分类
- `skills://numerical_issues/dim_imbalance` - 维度不平衡案例
- `skills://numerical_issues/nan_inf` - NaN/Inf问题案例
- `skills://common_fixes/recipes` - 修复方案

### Prompts（提示词）

- `static_check` - 指导Agent如何进行静态代码检查的完整提示词

## 使用方式

### 1. 配置MCP Client

在支持MCP的客户端（如Claude Desktop）中添加配置：

```json
{
  "mcpServers": {
    "training-profiler": {
      "command": "python",
      "args": ["/path/to/mcp_server/server.py"]
    }
  }
}
```

### 2. 与Agent交互

用户可以通过自然语言让Agent进行检查：

**用户**："请检查我的模型代码 model.py 的混合精度风险"

**Agent的工作流程**：
1. 调用 `check_file` 读取 model.py
2. 调用 `read_resource` 加载 unsafe_ops 知识
3. 调用 `search_pattern` 搜索各种风险模式
4. 调用 `analyze_code` 综合分析
5. 调用 `get_fix_suggestion` 获取修复方案
6. 整理输出报告给用户

### 3. 示例对话

**用户**："这段代码有什么问题？"
```python
class Model(nn.Module):
    def forward(self, x):
        x = x.half()
        return self.layer(x)
```

**Agent**：
1. 调用 `analyze_code` 分析代码
2. 发现问题：第3行使用了`.half()`
3. 调用 `get_fix_suggestion` 获取修复方案
4. 返回给用户：

```
## 问题 1

**位置**：第 3 行
**类型**：manual_cast
**严重级别**：HIGH
**问题描述**：使用了手动类型转换 .half()

**风险解释**：
手动.half()会绕过PyTorch AMP的自动类型提升机制，可能导致：
1. 与其他FP32组件的类型冲突
2. 梯度缩放失效
3. 无法享受AMP的数值保护

**修复建议**：
删除 .half() 调用，改为使用 autocast：

```python
def forward(self, x):
    with autocast():  # 添加这一行
        return self.layer(x)
```

**参考文档**：
skills://unsafe_ops/pytorch_amp
```

## 目录结构

```
mcp_server/
├── server.py                    # MCP Server主程序
├── skills/                      # 知识库（JSON格式）
│   ├── unsafe_ops/
│   │   └── pytorch_amp.json    # PyTorch AMP算子分类
│   ├── numerical_issues/
│   │   ├── dim_imbalance.json  # 维度不平衡案例
│   │   └── nan_inf.json        # NaN/Inf案例
│   └── common_fixes/
│       └── recipes.json        # 修复方案
├── prompts/
│   └── static_check.md         # Agent提示词
└── tests/
    └── test_static_check.py    # 单元测试
```

## 开发

### 运行测试

```bash
cd mcp_server
pytest tests/ -v
```

### 添加新的Skill

1. 在 `skills/{category}/` 下创建 `{name}.json`
2. 在 `server.py` 的 `list_resources` 中注册
3. 在 `server.py` 的 `read_resource` 中添加读取逻辑

### 添加新的检查模式

1. 在 `server.py` 的 `handle_search_pattern` 中添加新的模式识别
2. 在 `handle_analyze_code` 中添加对应的检查逻辑
3. 在 `handle_get_fix_suggestion` 中添加修复建议

## 优势

相比传统代码检查工具：

1. **自然语言交互**：用户不需要学习工具命令，直接对话即可
2. **上下文理解**：Agent可以结合用户的问题背景给出建议
3. **知识可扩展**：通过skills目录持续积累案例和知识
4. **多轮对话**：可以追问、澄清、深入分析

## 后续扩展

- 添加更多数值问题案例（如梯度爆炸/消失）
- 添加框架特定知识（Megatron/DeepSpeed的FP8注意事项）
- 集成P1的监控功能（通过MCP提供训练时数据）
