"""
MoE模型配置和集成建议

为self-ensemble系统提供MoE模型支持，
重点关注资源友好的选择和实际应用建议。
"""


def is_moe_model(model_name):
    """检查是否是MoE模型"""
    moe_indicators = ["moe", "mixtral"]
    return any(indicator in model_name.lower() for indicator in moe_indicators)


def get_moe_model_info(model_name):
    """获取MoE模型的详细信息"""
    moe_info = {
        "qwen1.5_moe_a2.7b": {
            "total_params": "14.3B",
            "active_params": "2.7B",
            "memory_requirement": "~6GB",
            "speed": "快",
            "quality": "优于3B密集模型",
            "recommended_for": "资源有限但要求高质量"
        },
        "qwen1.5_moe_a2.7b_chat": {
            "total_params": "14.3B",
            "active_params": "2.7B",
            "memory_requirement": "~6GB",
            "speed": "快",
            "quality": "优于3B密集模型，专门优化对话",
            "recommended_for": "QA任务和对话应用"
        }
    }
    return moe_info.get(model_name, {})


# MoE模型在self-ensemble中的优势分析
MOE_IN_SELF_ENSEMBLE_ANALYSIS = """
🧠 MoE模型在Self-Ensemble中的深度分析

## 🎯 为什么MoE特别适合Self-Ensemble任务？

### 1. **双重多样性机制**
- **内在多样性**: MoE内部有多个专家，每个处理不同类型的输入
- **外在多样性**: Self-ensemble提供多种改写和提示
- **协同效应**: 两种多样性相互增强，覆盖更广的答案空间

### 2. **专家激活模式与改写类型的匹配**
```
改写类型1: "What is the capital of France?" 
→ 激活专家A (地理知识专家)

改写类型2: "Which city serves as France's capital?"
→ 激活专家B (语言理解专家) 

改写类型3: "France's capital city is?"
→ 激活专家C (补全专家)
```

### 3. **自然的权重分配**
- MoE的门控机制自动为不同专家分配权重
- 可以减少人工设计权重的复杂性
- 动态适应不同类型的问题

## 📊 性能预期

### 相比传统方法的优势：
1. **质量提升**: 2.7B MoE ≈ 7B 密集模型的效果
2. **效率提升**: 只激活部分专家，推理速度快
3. **鲁棒性**: 不同专家处理边缘情况，减少失误

### 可能的劣势：
1. **复杂性**: 模型行为更难预测和调试
2. **专家不平衡**: 某些专家可能过度依赖
3. **内存碎片**: 多专家结构可能导致内存使用不够高效

## 🔧 集成策略建议

### Strategy 1: MoE作为基础模型
```python
# 用MoE替换现有的密集模型
python generate.py --model qwen1.5_moe_a2.7b_chat --method avg
```
**优势**: 简单直接，现有代码无需修改
**适用**: 快速验证MoE效果

### Strategy 2: 利用MoE的专家信息
```python
# 可能的未来扩展：根据专家激活模式调整集成权重
# 当前代码暂不支持，需要深度定制
```
**优势**: 充分利用MoE特性
**劣势**: 需要大量开发工作

### Strategy 3: 混合架构
```python
# 同时使用MoE和密集模型
models = ["qwen1.5_moe_a2.7b_chat", "qwen3_4b_it", "llama3.2_3b_it"]
# 在模型级别进行集成
```
**优势**: 结合两种架构的优点
**劣势**: 资源需求增加

## 💡 实际使用建议

### 对于您的资源情况：
1. **首选**: qwen1.5_moe_a2.7b_chat
   - 6GB显存可以轻松运行
   - 对话优化，适合QA任务
   - 比3B密集模型效果更好

2. **测试流程**:
   ```bash
   # 步骤1: 基础测试
   python generate.py --model qwen1.5_moe_a2.7b_chat --method per_prompt
   
   # 步骤2: 集成测试  
   python generate.py --model qwen1.5_moe_a2.7b_chat --method weighted_avg
   
   # 步骤3: 效果对比
   # 与现有最好模型对比准确率
   ```

3. **性能监控**:
   - 观察内存使用情况
   - 对比推理速度
   - 评估答案质量

## 🎮 调试和优化

### 调试MoE特有问题：
1. **专家利用率不均**: 
   - 症状：某些专家从不激活
   - 解决：调整temperature和top_p参数

2. **内存峰值**:
   - 症状：间歇性OOM
   - 解决：减少batch_size，使用gradient_checkpointing

3. **推理不稳定**:
   - 症状：相同输入产生不同输出
   - 解决：设置固定随机种子，降低temperature

### 优化建议：
```python
# MoE专用的生成参数
moe_config = {
    "temperature": 0.6,    # 稍低，减少随机性
    "top_p": 0.9,         # 较高，保持多样性
    "do_sample": True,    # 利用MoE的多样性
    "num_beams": 1,       # MoE已有内在多样性，beam search可能冗余
}
```
"""


def should_use_moe_for_task(task_type, resource_level):
    """
    判断是否应该为特定任务使用MoE模型

    Args:
        task_type: "qa", "generation", "chat", "reasoning" 
        resource_level: "low", "medium", "high"

    Returns:
        dict: 建议信息
    """

    recommendations = {
        ("qa", "low"): {
            "use_moe": True,
            "model": "qwen1.5_moe_a2.7b_chat",
            "reason": "QA任务受益于MoE的知识多样性，2.7B MoE在有限资源下效果最佳"
        },
        ("qa", "medium"): {
            "use_moe": True,
            "model": "qwen1.5_moe_a2.7b_chat",
            "reason": "可以考虑更大的MoE模型，但2.7B已经足够好"
        },
        ("generation", "low"): {
            "use_moe": False,
            "model": "qwen3_1.7b_it",
            "reason": "创意生成可能不需要MoE的复杂性，密集模型更可预测"
        },
        ("chat", "low"): {
            "use_moe": True,
            "model": "qwen1.5_moe_a2.7b_chat",
            "reason": "对话任务需要处理多种类型输入，MoE的专家机制很有帮助"
        }
    }

    key = (task_type, resource_level)
    return recommendations.get(key, {
        "use_moe": False,
        "model": "qwen3_4b_it",
        "reason": "默认选择平衡的密集模型"
    })


if __name__ == "__main__":
    print(MOE_IN_SELF_ENSEMBLE_ANALYSIS)

    # 示例：为QA任务在低资源环境下选择模型
    rec = should_use_moe_for_task("qa", "low")
    print(f"\nRecommendation:")
    print(f"Use MoE: {rec['use_moe']}")
    print(f"Recommended model: {rec['model']}")
    print(f"Reason: {rec['reason']}")
