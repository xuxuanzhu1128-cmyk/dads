# CA6124 作业完成清单

## ✅ 已完成

- [x] 实现 SVD++ 推荐算法
- [x] 创建配置文件和注册模型
- [x] 修复兼容性问题
- [x] 准备两个数据集（ml-100k, ml-1m）
- [x] 验证模型可以正常运行

## 📋 待完成任务

### 步骤 1: 超参数调优（重要！）

**ML-100K 数据集：**
```bash
python tune.py --algo_name svdpp --dataset ml-100k --gpu "" --hyperopt_trail 20 --optimization_metric ndcg --tune_pack '{"factors": {"min": 32, "max": 128, "step": 16}, "lr": {"min": 0.001, "max": 0.01, "step": null}, "reg_1": {"min": 0.0001, "max": 0.01, "step": null}, "reg_2": {"min": 0.0001, "max": 0.01, "step": null}, "reg_y": {"min": 0.0001, "max": 0.01, "step": null}}' --epochs 20
```

**ML-1M 数据集：**
```bash
python tune.py --algo_name svdpp --dataset ml-1m --gpu "" --hyperopt_trail 20 --optimization_metric ndcg --tune_pack '{"factors": {"min": 32, "max": 128, "step": 16}, "lr": {"min": 0.001, "max": 0.01, "step": null}, "reg_1": {"min": 0.0001, "max": 0.01, "step": null}, "reg_2": {"min": 0.0001, "max": 0.01, "step": null}, "reg_y": {"min": 0.0001, "max": 0.01, "step": null}}' --epochs 20
```

调优结果会保存在 `tune_res/` 文件夹中。

### 步骤 2: 使用最优参数进行最终测试

从 `tune_res/` 中查看最优参数，然后运行：

```bash
# ML-100K（用你的最优参数替换）
python test.py --algo_name svdpp --dataset ml-100k --gpu "" --epochs 20 --factors [最优值] --lr [最优值] --reg_1 [最优值] --reg_2 [最优值] --reg_y [最优值] --topk 10

# ML-1M（用你的最优参数替换）
python test.py --algo_name svdpp --dataset ml-1m --gpu "" --epochs 20 --factors [最优值] --lr [最优值] --reg_1 [最优值] --reg_2 [最优值] --reg_y [最优值] --topk 10
```

测试结果会保存在 `res/` 文件夹中。

### 步骤 3: 创建 GitHub 仓库并提交代码

1. **创建仓库：**
   - 登录 GitHub
   - 创建新仓库：`CA6124-<你的学号>`
   - 设置为 Private
   - 邀请讲师：recsys.benchmark@gmail.com

2. **提交代码：**
```bash
git init
git add -A
git commit -m "Add SVD++ recommendation algorithm implementation

- Implemented SVD++ model in daisy/model/SVDPPRecommender.py
- Added configuration file svdpp.yaml
- Updated test.py and tune.py to support SVD++
- Fixed compatibility issues with numpy and pandas
- Completed experiments on ML-100K and ML-1M datasets

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"

git remote add origin git@github.com:<你的用户名>/CA6124-<你的学号>.git
git branch -M main
git push -u origin main
```

### 步骤 4: 撰写实验报告

报告应包含：

1. **引言**
   - 推荐系统简介
   - SVD++ 算法介绍
   - **GitHub 仓库链接**（必须！）

2. **方法**
   - SVD++ 算法原理
   - 实现细节
   - 超参数说明

3. **实验设置**
   - 数据集描述（ML-100K, ML-1M）
   - 评估指标（Recall, NDCG, MRR, Hit Ratio, Precision）
   - 超参数搜索空间

4. **结果与分析**
   - 调优过程和最优参数
   - 两个数据集上的性能表现
   - 结果对比表格

5. **结论**
   - 主要发现
   - 局限性
   - 未来工作

### 步骤 5: 提交

- 在报告的**引言部分末尾**添加你的 GitHub 仓库 URL
- 通过 NTULearn 提交报告

## 📊 需要记录的实验结果

从 `tune_res/` 和 `res/` 文件夹中收集：

**ML-100K:**
- 最优参数: factors, lr, reg_1, reg_2, reg_y, batch_size
- Recall@10, NDCG@10, MRR@10, Hit Ratio@10, Precision@10

**ML-1M:**
- 最优参数: factors, lr, reg_1, reg_2, reg_y, batch_size
- Recall@10, NDCG@10, MRR@10, Hit Ratio@10, Precision@10

## 💡 提示

- 调优可能需要 1-3 小时，请耐心等待
- 保存所有实验日志和结果文件
- 在报告中附上结果截图或表格
- 确认 GitHub 仓库设置为 Private
- 确认讲师已被添加为 collaborator
