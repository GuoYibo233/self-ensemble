# Self-Ensemble 测试指南

## 🎯 测试层级

### Level 1: 基础功能测试 ✅
- [x] 环境依赖测试 (`test_environment.py`)
- [x] 配置模块测试 (`test_qwen3.py`, `test_moe.py`)  
- [x] 调试脚本测试 (`debug_generate.py`)

### Level 2: 模拟模型测试 ✅
- [x] 使用mock模型进行完整流程测试
- [x] 验证所有ensemble方法正常工作
- [x] 确认输出格式正确

### Level 3: 真实模型测试（需要下载模型）

#### 3.1 小模型测试
```bash
# 下载并测试最小的模型
huggingface-cli download Qwen/Qwen3-1.7B-Instruct

# 运行真实测试
python generate.py --model qwen3_1.7b_it --dataset mock --method per_prompt --max_samples 5
```

#### 3.2 MoE模型测试  
```bash
# 下载MoE模型（推荐）
huggingface-cli download Qwen/Qwen1.5-MoE-A2.7B-Chat

# 测试MoE效果
python generate.py --model qwen1.5_moe_a2.7b_chat --dataset mock --method avg --max_samples 10
```

#### 3.3 性能对比测试
```bash
# 运行性能基准测试
python gpu_benchmark.py

# 对比不同ensemble方法
python generate.py --model qwen3_1.7b_it --method per_prompt --output per_prompt_results.json
python generate.py --model qwen3_1.7b_it --method avg --output avg_results.json
python generate.py --model qwen3_1.7b_it --method weighted_avg --output weighted_results.json
```

## 🔧 当前可执行的测试

### 立即可运行的测试：
```bash
# 环境检查
python test_environment.py

# 配置验证
python test_qwen3.py --test-all
python test_moe.py

# 调试和流程验证
python debug_generate.py --method per_prompt
python debug_generate.py --method avg
python debug_generate.py --method weighted_avg
python debug_generate.py --method max
python debug_generate.py --method weighted_max

# 性能基准（不需要真实模型）
python gpu_benchmark.py
```

### 需要真实模型的测试：
```bash
# 这些需要先下载模型文件
python generate.py --model qwen3_1.7b_it --dataset your_dataset --method avg
python paraphrase.py --input_file data.json --model qwen3_1.7b_it
```

## 📊 测试结果验证

### Debug模式输出示例
✅ 应该看到详细的中文调试信息（按您要求保留）
✅ 每个步骤的tensor信息和处理过程
✅ 不同ensemble方法的对比结果

### 生产模式输出示例  
✅ 应该看到英文的状态信息
✅ 简洁的进度指示
✅ 最终结果的JSON格式输出

## 🚀 下一步建议

1. **立即测试**：运行所有debug脚本确保功能正常
2. **选择模型**：根据您的GPU内存选择合适的模型下载
3. **小批量测试**：使用少量数据验证真实模型效果  
4. **性能优化**：根据测试结果调整参数和方法
5. **批量处理**：确认无误后进行大规模处理

## 💡 故障排除

### 常见问题：
- **内存不足**：使用较小模型或减少batch_size
- **CUDA问题**：检查PyTorch CUDA安装
- **模型路径**：确认MODEL_PATHs中的路径正确
- **依赖缺失**：重新运行conda install

### 调试技巧：
- 使用debug_generate.py理解内部流程
- 检查generate.py的详细日志
- 对比mock和真实模型的输出差异