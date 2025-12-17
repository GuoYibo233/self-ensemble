# 项目结构说明


### 放在generate外的文件，除了那个看mask的都是你原来的
```
constants.py                      # 模型路径配置
dataset.py                        # 数据集加载器
paraphrase.py                     # Paraphrase生成
confidence.py                     # 置信度计算
utils.py                          # 工具函数
mask_visualization.py             # Attention mask可视化
```

### 配置
```
requirements.txt                  # Python依赖包列表
environment.yml                   # Conda环境配置（我windows上面用的）
environment_linux.yml             # Conda环境配置（Linux专用，这个我感觉没问题）
```


# Generation Scripts Overview

## Prompt格式说明

### 格式1: 标准格式（Baseline和Original）
```
[BOS] Instruction Few-shot Paraphrase1 [生成] [EOS]
```
- 每个paraphrase单独生成
- 包含完整的instruction和few-shot

### 格式2: FlexAttention拼接格式
```
[BOS] [Ins+FS+P1] [Ins+FS+P2] [Ins+FS+P3] [生成] [EOS]
```
- 每个paraphrase都包含完整的instruction和few-shot
- 多个paraphrase拼接在一起
- 编码阶段各paraphrase互相隔离
- 生成阶段融合所有paraphrase

### 格式3: 共享Prompt格式（MyriadLAMA专用）
```
[BOS] [共享Ins+FS] [P1] [P2] [P3] [生成] [EOS]
```
- Instruction和few-shot只出现一次（共享）
- 各paraphrase只包含问题部分
- 编码阶段各paraphrase互相隔离
- 生成阶段融合所有paraphrase

---

## 1. generate_baseline.py

- MyriadLAMA / WebQA
- 只有一个问题，不存在paraphrase

---

## 2. generate_original.py

- MyriadLAMA / WebQA
- 也是，只有一个问题
---

## 3. generate_flex_attention.py

- MyriadLAMA / WebQA
- prompt是第二种拼在一起的
- Flex attention

---

## 4. generate_myriadlama.py

- 仅MyriadLAMA
- 格式3（共享Prompt格式）
- 使用FlexAttention API

---

## 5. myriadlama_custom_attention_generate.py

- 仅MyriadLAMA
- 格式3（共享Prompt格式）
- 使用自定义attention mask
- 能修改position， 搜position_ids, 有函数 统一所有paraphrase的position embedding，使用padding对齐
