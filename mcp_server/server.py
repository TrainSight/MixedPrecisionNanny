"""
TrainingProfiler MCP Server
通过MCP协议提供混合精度训练的静态代码检查能力
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# MCP SDK
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent


# 加载skills知识库
SKILLS_DIR = Path(__file__).parent / "skills"


def load_skill(category: str, name: str) -> Dict[str, Any]:
    """加载指定skill"""
    skill_file = SKILLS_DIR / category / f"{name}.json"
    if skill_file.exists():
        with open(skill_file) as f:
            return json.load(f)
    return {}


def list_skills(category: Optional[str] = None) -> List[str]:
    """列出所有可用的skills"""
    if category:
        skill_dir = SKILLS_DIR / category
        if skill_dir.exists():
            return [f.stem for f in skill_dir.glob("*.json")]
        return []
    
    # 返回所有分类
    result = []
    for cat_dir in SKILLS_DIR.iterdir():
        if cat_dir.is_dir():
            result.append(f"{cat_dir.name}/*")
    return result


# 创建MCP Server
app = Server("mixed-precision-nanny")


@app.list_resources()
async def list_resources() -> List[Resource]:
    """列出所有可用的知识资源"""
    resources = []
    
    # Unsafe ops 列表
    resources.append(Resource(
        uri="skills://unsafe_ops/pytorch_amp",
        name="PyTorch AMP Unsafe Operations",
        mimeType="application/json",
        description="PyTorch AMP中需要注意的算子分类"
    ))
    
    # 数值问题案例
    resources.append(Resource(
        uri="skills://numerical_issues/dim_imbalance",
        name="维度不平衡案例",
        mimeType="application/json",
        description="维度不平衡导致的大数吃小数问题"
    ))
    
    resources.append(Resource(
        uri="skills://numerical_issues/nan_inf",
        name="NaN/Inf问题案例",
        mimeType="application/json",
        description="NaN和Inf的产生与检测"
    ))
    
    # 修复方案
    resources.append(Resource(
        uri="skills://common_fixes/recipes",
        name="常见修复方案",
        mimeType="application/json",
        description="混合精度问题的快速修复方案"
    ))
    
    return resources


@app.read_resource()
async def read_resource(uri: str) -> str:
    """读取指定资源"""
    if uri.startswith("skills://"):
        parts = uri.replace("skills://", "").split("/")
        if len(parts) >= 2:
            category = parts[0]
            name = parts[1]
            skill = load_skill(category, name)
            return json.dumps(skill, indent=2, ensure_ascii=False)
    
    return json.dumps({"error": "Resource not found"})


@app.list_tools()
async def list_tools() -> List[Tool]:
    """列出所有可用工具"""
    return [
        Tool(
            name="analyze_code",
            description="分析PyTorch模型代码，识别混合精度训练风险",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要分析的Python代码"
                    },
                    "focus_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "重点检查区域，如['manual_cast', 'loss_computation', 'reduce_ops']",
                        "default": ["all"]
                    }
                },
                "required": ["code"]
            }
        ),
        Tool(
            name="search_pattern",
            description="在代码中搜索特定风险模式",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "要搜索的代码"
                    },
                    "pattern_type": {
                        "type": "string",
                        "enum": ["half_call", "bce_loss", "autocast_scope", "reduce_op"],
                        "description": "要搜索的模式类型"
                    }
                },
                "required": ["code", "pattern_type"]
            }
        ),
        Tool(
            name="get_fix_suggestion",
            description="获取特定问题的修复建议",
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_type": {
                        "type": "string",
                        "enum": ["manual_half", "bce_loss", "reduce_precision", "dim_imbalance"],
                        "description": "问题类型"
                    }
                },
                "required": ["issue_type"]
            }
        ),
        Tool(
            name="check_file",
            description="读取并分析Python文件",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Python文件的绝对路径"
                    }
                },
                "required": ["file_path"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    """调用工具"""
    
    if name == "search_pattern":
        return await handle_search_pattern(arguments)
    elif name == "get_fix_suggestion":
        return await handle_get_fix_suggestion(arguments)
    elif name == "check_file":
        return await handle_check_file(arguments)
    elif name == "analyze_code":
        return await handle_analyze_code(arguments)
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def handle_search_pattern(arguments: dict) -> List[TextContent]:
    """处理模式搜索请求 - 返回结构化结果供Agent分析"""
    code = arguments.get("code", "")
    pattern_type = arguments.get("pattern_type", "")
    
    results = []
    lines = code.split("\n")
    
    if pattern_type == "half_call":
        # 搜索 .half() 或 .bfloat16()
        for i, line in enumerate(lines, 1):
            if ".half()" in line or ".bfloat16()" in line:
                results.append({
                    "line": i,
                    "content": line.strip(),
                    "risk": "绕过AMP保护机制"
                })
    
    elif pattern_type == "bce_loss":
        # 搜索 binary_cross_entropy
        for i, line in enumerate(lines, 1):
            if "binary_cross_entropy" in line and "with_logits" not in line:
                results.append({
                    "line": i,
                    "content": line.strip(),
                    "risk": "在AMP中不稳定，会导致NaN"
                })
    
    elif pattern_type == "autocast_scope":
        # 分析autocast作用域
        in_autocast = False
        autocast_start = 0
        for i, line in enumerate(lines, 1):
            if "with autocast" in line or "@torch.autocast" in line:
                in_autocast = True
                autocast_start = i
            elif in_autocast and line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                # 缩进结束，autocast块结束
                results.append({
                    "autocast_block": {"start": autocast_start, "end": i - 1}
                })
                in_autocast = False
    
    elif pattern_type == "reduce_op":
        # 搜索 reduce 操作
        reduce_ops = [".sum(", ".mean(", ".cumsum(", ".prod("]
        for i, line in enumerate(lines, 1):
            for op in reduce_ops:
                if op in line:
                    results.append({
                        "line": i,
                        "content": line.strip(),
                        "op": op.strip(".("),
                        "risk": "大数吃小数风险，建议确认数值范围"
                    })
    
    return [TextContent(
        type="text",
        text=json.dumps({
            "pattern_type": pattern_type,
            "matches": results,
            "count": len(results)
        }, indent=2, ensure_ascii=False)
    )]


async def handle_get_fix_suggestion(arguments: dict) -> List[TextContent]:
    """获取修复建议"""
    issue_type = arguments.get("issue_type", "")
    
    suggestions = {
        "manual_half": {
            "problem": "手动.half()会绕过AMP的自动类型提升",
            "fix": "删除.half()调用，使用with autocast()上下文",
            "example": {
                "before": "x = x.half()",
                "after": "# 删除这行，在forward中使用with autocast():"
            }
        },
        "bce_loss": {
            "problem": "binary_cross_entropy在AMP中会数值不稳定",
            "fix": "改用binary_cross_entropy_with_logits",
            "example": {
                "before": "F.binary_cross_entropy(pred, target)",
                "after": "F.binary_cross_entropy_with_logits(logits, target)"
            }
        },
        "reduce_precision": {
            "problem": "reduce操作可能丢失精度",
            "fix": "在reduce前转FP32",
            "example": {
                "before": "result = tensor.sum(dim=-1)",
                "after": "result = tensor.float().sum(dim=-1)"
            }
        },
        "dim_imbalance": {
            "problem": "tensor维度间数值范围差异大",
            "fix": "分别处理不同维度，或在reduce前归一化",
            "example": {
                "before": "loss = data.sum(dim=0)",
                "after": "loss = data.float().sum(dim=0)  # 或分维度处理"
            }
        }
    }
    
    suggestion = suggestions.get(issue_type, {"error": "Unknown issue type"})
    
    return [TextContent(
        type="text",
        text=json.dumps(suggestion, indent=2, ensure_ascii=False)
    )]


async def handle_check_file(arguments: dict) -> List[TextContent]:
    """检查文件"""
    file_path = arguments.get("file_path", "")
    
    if not os.path.exists(file_path):
        return [TextContent(type="text", text=f"File not found: {file_path}")]
    
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "file": file_path,
                "lines": len(code.split("\n")),
                "size_bytes": len(code),
                "code": code
            }, indent=2)
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Error reading file: {str(e)}")]


async def handle_analyze_code(arguments: dict) -> List[TextContent]:
    """综合分析代码 - 返回详细报告供Agent解读"""
    code = arguments.get("code", "")
    focus_areas = arguments.get("focus_areas", ["all"])
    
    # 加载知识库
    unsafe_ops = load_skill("unsafe_ops", "pytorch_amp")
    
    analysis = {
        "summary": {
            "total_lines": len(code.split("\n")),
            "risk_level": "unknown",
            "issue_count": 0
        },
        "issues": []
    }
    
    lines = code.split("\n")
    
    # 检查1: 手动类型转换
    if "all" in focus_areas or "manual_cast" in focus_areas:
        for i, line in enumerate(lines, 1):
            if ".half()" in line or ".bfloat16()" in line:
                analysis["issues"].append({
                    "line": i,
                    "type": "manual_cast",
                    "severity": "HIGH",
                    "content": line.strip(),
                    "description": "手动类型转换可能绕过AMP保护",
                    "fix_type": "manual_half"
                })
    
    # 检查2: 不安全的loss函数
    if "all" in focus_areas or "loss_computation" in focus_areas:
        for i, line in enumerate(lines, 1):
            if "binary_cross_entropy" in line and "with_logits" not in line:
                analysis["issues"].append({
                    "line": i,
                    "type": "unsafe_loss",
                    "severity": "HIGH",
                    "content": line.strip(),
                    "description": "binary_cross_entropy在AMP中不稳定",
                    "fix_type": "bce_loss"
                })
    
    # 检查3: Reduce操作
    if "all" in focus_areas or "reduce_ops" in focus_areas:
        reduce_ops = [".sum(", ".mean(", ".cumsum(", ".prod("]
        for i, line in enumerate(lines, 1):
            for op in reduce_ops:
                if op in line:
                    analysis["issues"].append({
                        "line": i,
                        "type": "reduce_op",
                        "severity": "MEDIUM",
                        "content": line.strip(),
                        "description": f"{op.strip('.(')}操作可能存在精度损失",
                        "fix_type": "reduce_precision"
                    })
    
    # 更新汇总
    analysis["summary"]["issue_count"] = len(analysis["issues"])
    high_count = sum(1 for i in analysis["issues"] if i["severity"] == "HIGH")
    
    if high_count > 0:
        analysis["summary"]["risk_level"] = "HIGH"
    elif analysis["summary"]["issue_count"] > 0:
        analysis["summary"]["risk_level"] = "MEDIUM"
    else:
        analysis["summary"]["risk_level"] = "LOW"
    
    return [TextContent(
        type="text",
        text=json.dumps(analysis, indent=2, ensure_ascii=False)
    )]


if __name__ == "__main__":
    # 运行MCP Server
    import asyncio
    from mcp.server.stdio import stdio_server
    
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    
    asyncio.run(main())
