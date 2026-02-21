"""
MCP Server 静态代码检查单元测试

测试内容：
1. 模式搜索功能
2. 修复建议功能
3. 综合分析功能
4. 知识库加载
"""
import pytest
import json
import sys
from pathlib import Path

# 添加server到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import (
    load_skill,
    list_skills,
    handle_search_pattern,
    handle_get_fix_suggestion,
    handle_analyze_code,
    handle_check_file
)


# ==================== 测试数据 ====================

TEST_CODE_MANUAL_HALF = """
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        x = x.half()  # 第7行：手动转换
        return x * 2
"""

TEST_CODE_BCE_LOSS = """
import torch.nn.functional as F

# 使用不安全的loss
loss = F.binary_cross_entropy(pred, target)

# 安全的用法
loss = F.binary_cross_entropy_with_logits(logits, target)
"""

TEST_CODE_REDUCE = """
import torch

# Reduce操作
total = tensor.sum(dim=-1)
avg = tensor.mean(dim=0)
"""

TEST_CODE_CLEAN = """
import torch
from torch.cuda.amp import autocast

class Model(torch.nn.Module):
    def forward(self, x):
        with autocast():
            return self.layer(x)
"""


# ==================== 测试用例 ====================

@pytest.mark.asyncio
async def test_search_pattern_half_call():
    """测试搜索.half()调用"""
    result = await handle_search_pattern({
        "code": TEST_CODE_MANUAL_HALF,
        "pattern_type": "half_call"
    })
    
    data = json.loads(result[0].text)
    assert data["pattern_type"] == "half_call"
    assert data["count"] == 1
    assert data["matches"][0]["line"] == 7
    assert ".half()" in data["matches"][0]["content"]


@pytest.mark.asyncio
async def test_search_pattern_bce_loss():
    """测试搜索不安全的BCE loss"""
    result = await handle_search_pattern({
        "code": TEST_CODE_BCE_LOSS,
        "pattern_type": "bce_loss"
    })
    
    data = json.loads(result[0].text)
    assert data["pattern_type"] == "bce_loss"
    assert data["count"] == 1
    # 应该只匹配非with_logits版本
    assert "with_logits" not in data["matches"][0]["content"]


@pytest.mark.asyncio
async def test_search_pattern_reduce_op():
    """测试搜索reduce操作"""
    result = await handle_search_pattern({
        "code": TEST_CODE_REDUCE,
        "pattern_type": "reduce_op"
    })
    
    data = json.loads(result[0].text)
    assert data["pattern_type"] == "reduce_op"
    assert data["count"] == 2  # sum和mean
    
    ops = [m["op"] for m in data["matches"]]
    assert "sum" in ops
    assert "mean" in ops


@pytest.mark.asyncio
async def test_get_fix_suggestion_manual_half():
    """测试获取手动类型转换的修复建议"""
    result = await handle_get_fix_suggestion({
        "issue_type": "manual_half"
    })
    
    data = json.loads(result[0].text)
    assert "problem" in data
    assert "fix" in data
    assert "example" in data
    assert ".half()" in data["example"]["before"]


@pytest.mark.asyncio
async def test_get_fix_suggestion_bce_loss():
    """测试获取BCE loss的修复建议"""
    result = await handle_get_fix_suggestion({
        "issue_type": "bce_loss"
    })
    
    data = json.loads(result[0].text)
    assert "with_logits" in data["example"]["after"]


@pytest.mark.asyncio
async def test_analyze_code_with_issues():
    """测试分析包含问题的代码"""
    code = TEST_CODE_MANUAL_HALF + TEST_CODE_BCE_LOSS
    
    result = await handle_analyze_code({
        "code": code,
        "focus_areas": ["all"]
    })
    
    data = json.loads(result[0].text)
    assert data["summary"]["issue_count"] >= 2
    assert data["summary"]["risk_level"] == "HIGH"
    
    # 检查是否识别到了具体问题
    issue_types = [i["type"] for i in data["issues"]]
    assert "manual_cast" in issue_types
    assert "unsafe_loss" in issue_types


@pytest.mark.asyncio
async def test_analyze_code_clean():
    """测试分析干净的代码"""
    result = await handle_analyze_code({
        "code": TEST_CODE_CLEAN,
        "focus_areas": ["all"]
    })
    
    data = json.loads(result[0].text)
    assert data["summary"]["issue_count"] == 0
    assert data["summary"]["risk_level"] == "LOW"


@pytest.mark.asyncio
async def test_analyze_code_focus_areas():
    """测试指定检查区域"""
    code = TEST_CODE_MANUAL_HALF + TEST_CODE_REDUCE
    
    # 只检查manual_cast
    result = await handle_analyze_code({
        "code": code,
        "focus_areas": ["manual_cast"]
    })
    
    data = json.loads(result[0].text)
    issue_types = [i["type"] for i in data["issues"]]
    assert "manual_cast" in issue_types
    # reduce_op不应该被检查
    assert "reduce_op" not in issue_types


def test_load_skill():
    """测试加载skill"""
    skill = load_skill("unsafe_ops", "pytorch_amp")
    assert "categories" in skill
    assert "force_fp32" in skill["categories"]


def test_list_skills():
    """测试列出skills"""
    skills = list_skills("unsafe_ops")
    assert "pytorch_amp" in skills


# ==================== 集成测试 ====================

@pytest.mark.asyncio
async def test_full_workflow():
    """
    完整工作流程测试：
    1. 加载知识库
    2. 搜索风险模式
    3. 综合分析
    4. 获取修复建议
    """
    # 1. 加载知识
    skill = load_skill("unsafe_ops", "pytorch_amp")
    assert skill is not None
    
    # 2. 搜索模式
    search_result = await handle_search_pattern({
        "code": TEST_CODE_MANUAL_HALF,
        "pattern_type": "half_call"
    })
    search_data = json.loads(search_result[0].text)
    assert search_data["count"] > 0
    
    # 3. 综合分析
    analyze_result = await handle_analyze_code({
        "code": TEST_CODE_MANUAL_HALF,
        "focus_areas": ["all"]
    })
    analyze_data = json.loads(analyze_result[0].text)
    assert analyze_data["summary"]["issue_count"] > 0
    
    # 4. 获取修复建议
    fix_result = await handle_get_fix_suggestion({
        "issue_type": analyze_data["issues"][0]["fix_type"]
    })
    fix_data = json.loads(fix_result[0].text)
    assert "fix" in fix_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
