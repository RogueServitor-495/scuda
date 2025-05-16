import torch

def test_divide_basic():
    """测试基本的除法操作"""
    torch.manual_seed(42)
    
    tp = torch.bfloat16
    
    # 创建两个张量
    a = torch.tensor([4.0, 9.0, 16.0],dtype=tp)
    b = torch.tensor([2.0, 3.0, 4.0],dtype=tp)
    
    
    
    # 使用torch.div
    result_div = torch.div(a, b)
    # 使用/运算符
    result_slash = a / b
    
    # 预期结果
    expected = torch.tensor([2.0, 3.0, 4.0])
    
    print(result_div)
    print(result_slash)

test_divide_basic()
