"""
极简模型代码检查
用法：
    python check_model.py your_model_file.py
"""
import ast
import sys


def check_file(filepath):
    """检查Python文件中的混合精度风险"""
    
    with open(filepath, 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    issues = []
    
    for node in ast.walk(tree):
        # 检查1：手动.half()调用
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'half':
                issues.append({
                    'line': node.lineno,
                    'type': 'manual_half',
                    'msg': "手动.half()可能跳过AMP保护，建议使用autocast"
                })
        
        # 检查2：softmax/log_softmax没有dtype参数
        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
            
            if func_name in ['softmax', 'log_softmax']:
                has_dtype = any(kw.arg == 'dtype' for kw in node.keywords)
                if not has_dtype:
                    issues.append({
                        'line': node.lineno,
                        'type': 'softmax_without_dtype',
                        'msg': f"{func_name}未指定dtype，在FP16下可能数值不稳定"
                    })
        
        # 检查3：binary_cross_entropy（不建议在AMP中使用）
        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
            
            if func_name in ['binary_cross_entropy', 'BCELoss']:
                issues.append({
                    'line': node.lineno,
                    'type': 'unsafe_loss',
                    'msg': "binary_cross_entropy在AMP中可能不稳定，建议用binary_cross_entropy_with_logits"
                })
        
        # 检查4：LayerNorm/BatchNorm后没有数值检查（粗略检查）
        if isinstance(node, ast.Call):
            func_name = None
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
            
            if func_name in ['LayerNorm', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d']:
                issues.append({
                    'line': node.lineno,
                    'type': 'normalization_layer',
                    'msg': f"{func_name}在混合精度中通常是安全的，但需确保输入范围合理"
                })
    
    return issues


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_model.py <model_file.py>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    issues = check_file(filepath)
    
    if not issues:
        print("✓ 未发现明显风险")
        return
    
    print(f"发现 {len(issues)} 个潜在问题：\n")
    
    for issue in issues:
        print(f"Line {issue['line']}: {issue['type']}")
        print(f"  -> {issue['msg']}\n")
    
    # 统计
    critical = sum(1 for i in issues if i['type'] in ['manual_half', 'unsafe_loss'])
    warning = len(issues) - critical
    
    print(f"统计: {critical} 个严重, {warning} 个警告")
    
    if critical > 0:
        print("\n建议：修复严重问题后再开始训练")
        sys.exit(1)


if __name__ == '__main__':
    main()
