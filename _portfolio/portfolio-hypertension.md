---
title: "社区高血压筛查对血压影响的断点回归分析"
excerpt: "基于CLHLS纵向数据的因果推断与机器学习预测分析<br/><img src='/images/rdd_hypertension_thumbnail.png'>"
collection: portfolio
---

## 项目概述

本项目综合运用**断点回归设计（RDD）**和**机器学习预测模型**，评估社区高血压筛查项目对血压控制的因果效应。基于中国老年健康影响因素跟踪调查（CLHLS）2011-2014年纵向数据（n=3,898）。

---

## 第一部分：数据处理与描述性分析

### 1.1 数据来源与样本筛选

**数据来源**：中国老年健康影响因素跟踪调查（CLHLS）2011-2018年纵向数据集

**样本筛选标准**：
- 2014年存活
- 基线年龄 ≥ 65岁
- 血压测量数据完整
- 排除既往确诊高血压患者

```python
# 样本筛选流程
df_14 = df[df['dth11_14'] == 0].copy()  # 2014年存活
df_raw = df_14[df_14['trueage'] >= 65].copy()  # 年龄≥65岁

# 血压变量生成
df_raw['sys11'] = df_raw[['g511', 'g521']].mean(axis=1)  # 基线收缩压
df_raw['dias11'] = df_raw[['g512', 'g522']].mean(axis=1)  # 基线舒张压
df_raw['sys14'] = df_raw[['g511_14', 'g521_14']].mean(axis=1)  # 随访收缩压
df_raw['dias14'] = df_raw[['g512_14', 'g522_14']].mean(axis=1)  # 随访舒张压
```

### 1.2 样本基线特征（表1）

| 特征 | 全样本 (n=3898) | 收缩压带宽内 (n=919) | 舒张压带宽内 (n=878) |
|------|-----------------|---------------------|---------------------|
| **人口学特征** |  |  |  |
| 平均年龄（岁） | 83.1 (10.6) | 82.8 (10.3) | 83.0 (11.0) |
| 男性比例 (%) | 47.8 | 46.2 | 50.0 |
| 城镇居民 (%) | 14.6 | 13.2 | 11.3 |
| 已婚 (%) | 44.9 | 43.0 | 45.6 |
| **社会经济特征** |  |  |  |
| 受教育年限 | 2.4 (3.4) | 2.3 (3.4) | 2.5 (3.4) |
| **健康行为** |  |  |  |
| 锻炼 (%) | 36.9 | 34.2 | 37.1 |
| 吸烟 (%) | 21.0 | 20.2 | 20.2 |
| 饮酒 (%) | 20.5 | 21.1 | 21.9 |

---

## 第二部分：断点回归设计（RDD）分析

### 2.1 研究设计

断点回归利用干预分配在阈值处的不连续性来识别因果效应：

| 要素 | 说明 |
|------|------|
| **驱动变量** | 2011-12年基线血压值 |
| **断点阈值** | 收缩压140 mmHg / 舒张压90 mmHg |
| **处理** | 超过阈值者接受健康指导 |
| **结果变量** | 2014年血压值 |
| **最优带宽** | 收缩压134-146 mmHg / 舒张压84-96 mmHg |

### 2.2 操纵性检验（图1）

McCrary密度检验用于检测个体是否在阈值处人为操纵血压测量值：

```python
from rddensity import rddensity

# 收缩压操纵性检验
rdd_sbp = rddensity(X=sbp_data.values, c=140)
print(f"T统计量: {rdd_sbp.hat['T']:.3f}")
print(f"P值: {rdd_sbp.hat['pv']:.3f}")
# 结果: P > 0.05，无操纵证据 ✓
```

**结果**：阈值处密度平滑过渡，无显著聚集，支持RDD设计有效性。

![密度分布图](/images/figure1_density.png)
*图1：基线血压在断点阈值处的密度分布*

### 2.3 协变量均衡性检验（表2）

断点两侧协变量应保持均衡以支持RDD有效性：

| 特征 | <140 mmHg | ≥140 mmHg | 差异 |
|------|-----------|-----------|------|
| 年龄 | 82.5 | 83.0 | 0.5 |
| 男性 (%) | 45.3 | 46.9 | 1.6 |
| 教育年限 | 2.3 | 2.4 | 0.1 |
| 锻炼 (%) | 33.8 | 34.5 | 0.7 |

*注：P<0.05水平下无显著差异*

### 2.4 断点回归可视化（图2）

![RDD图](/images/figure2_rdd_plot.png)
*图2：最优带宽内2014年血压与基线血压的关系*

在140 mmHg阈值处可见明显的向下跳跃，表明存在处理效应。

### 2.5 断点回归估计结果（表3）

```python
def rdd_regression(df, outcome_var, running_var, cutoff, bandwidth, covariates=None):
    """
    局部线性回归 + 三角核函数
    Y = α0 + α1*Above + α2*(BP - cutoff) + α3*Above*(BP - cutoff) + X*κ + ε
    """
    # 三角核权重
    weights = triangular_kernel(df['centered'], bandwidth)
    
    # 加权最小二乘
    model = sm.WLS(y, X, weights=weights)
    results = model.fit()
    return results.params['above'], results.conf_int()
```

**主要结果：**

| 血压效应 | 无协变量 | 控制全部协变量 |
|----------|----------|----------------|
| **收缩压** |  |  |
| 局部线性回归 | -6.4 (-10.8, -2.0)** | -8.5 (-13.0, -3.9)*** |
| 局部二次回归 | -10.2 (-18.1, -2.3)* | -15.7 (-23.7, -7.7)*** |
| **舒张压** |  |  |
| 局部线性回归 | -2.1 (-5.2, 0.9) | -2.1 (-5.4, 1.2) |
| 局部二次回归 | -0.9 (-6.8, 4.9) | -2.6 (-8.8, 3.5) |

*注：\* P<0.05, \*\* P<0.01, \*\*\* P<0.001*

**核心发现**：社区高血压筛查使收缩压在约3年后显著降低 **8.5-15.7 mmHg**。

### 2.6 带宽敏感性分析（图3）

![带宽敏感性](/images/figure3_bandwidth_sensitivity.png)
*图3：不同带宽下的处理效应估计*

结果在不同带宽选择（1-9 mmHg）下保持稳健。

---

## 第三部分：机器学习预测模型

### 3.1 特征工程

```python
# 创建干预变量
df['above'] = ((df['sbp_2011'] >= 140) | (df['dbp_2011'] >= 90)).astype(int)

# 交互项
df['above_sbp_interact'] = df['above'] * (df['sbp_2011'] - 140)

# 饮食健康得分（健康食品 - 不健康食品）
df['diet_score'] = (df['fruit_freq'] + df['veg_freq']) - (df['slatveg_freq'] + df['sugar_freq'])

# 特征列表
features = [
    'sbp_2011', 'dbp_2011', 'above',  # 血压与干预
    'age', 'sex', 'residence', 'marital', 'child',  # 人口统计学
    'edu_years', 'eco_status',  # 社会经济
    'exercise', 'smoke', 'drink',  # 健康行为
    'fruit_freq', 'veg_freq', 'meat_freq', 'slatveg_freq', 'sugar_freq'  # 饮食
]
```

### 3.2 模型构建

1. **线性回归** - 基准模型
2. **岭回归** - L2正则化
3. **随机森林** - 集成树模型
4. **梯度提升** - 序列集成

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5)
}
```

### 3.3 模型性能对比

| 模型 | RMSE (mmHg) | MAE (mmHg) | R² | 临床准确率* |
|------|-------------|------------|-----|------------|
| **收缩压** |  |  |  |  |
| 线性回归 | 58.43 | 18.89 | -0.002 | 41.9% |
| 岭回归 | 58.43 | 18.89 | -0.002 | 41.9% |
| 随机森林 | 60.71 | 21.12 | -0.082 | 43.4% |
| 梯度提升 | 63.41 | 21.98 | -0.180 | 40.4% |
| **舒张压** |  |  |  |  |
| 线性回归 | 60.15 | 13.90 | -0.001 | 58.6% |
| 岭回归 | 60.15 | 13.90 | -0.001 | 58.7% |
| 随机森林 | 63.99 | 16.97 | -0.133 | 58.3% |
| 梯度提升 | 67.36 | 18.25 | -0.255 | 56.9% |

*临床准确率：预测值与实际值相差10 mmHg以内的比例*

### 3.4 特征重要性分析

**收缩压预测前10重要特征（梯度提升模型）：**

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | 基线舒张压 | 0.1587 |
| 2 | 饮食评分 | 0.1522 |
| 3 | 年龄 | 0.1296 |
| 4 | 基线收缩压 | 0.0814 |
| 5 | 子女数量 | 0.0771 |
| 6 | 受教育年限 | 0.0615 |
| 7 | 腌制蔬菜频率 | 0.0550 |
| 8 | 水果摄入频率 | 0.0437 |
| 9 | 经济状况 | 0.0413 |
| 10 | 肉类摄入频率 | 0.0390 |

### 3.5 模型可视化

![模型可视化](/images/fig_prediction_model_sbp.png)
*图4：(a) 真实值vs预测值散点图；(b) 特征重要性；(c) 干预效应分布；(d) 模型性能雷达图*

---

## 第四部分：主要发现与结论

### 主要结论

1. **✓ 收缩压显著降低**：社区筛查使收缩压降低 **8.5-15.7 mmHg**（P<0.001）

2. **✗ 舒张压无显著变化**：效应接近零且不显著（P=0.20-0.40）

3. **✓ 结果稳健**：不同带宽和模型设定下结果一致

4. **⚠ 预测局限性**：低R²值表明血压变化受多种未测量因素影响

### 政策启示

- 支持继续推广社区高血压筛查项目
- 重点关注收缩压偏高的老年人群
- 加强对筛查人群的后续健康管理

### 研究局限

- 样本主要为老年人群（≥65岁）
- 缺乏详细干预信息（用药、生活方式改变）
- 血压测量可能存在误差

---

## 技术实现

### 使用工具

```python
# 数据处理
import pandas as pd
import numpy as np
import pyreadstat  # 读取SPSS数据

# 统计分析
import statsmodels.api as sm
from scipy import stats
from rddensity import rddensity  # McCrary检验

# 机器学习
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 文档生成
# LaTeX (XeLaTeX + ctexart 中文支持)
```

### 项目结构

```
论文复现/
├── dofiles/
│   └── 数据分析.ipynb          # 主分析代码（2700+行）
├── raw_data/
│   └── clhls_*.sav             # CLHLS纵向数据
├── working_data/
│   ├── figure1_density.pdf     # 操纵性检验图
│   ├── figure2_rdd_plot.pdf    # RDD可视化
│   ├── figure3_bandwidth_sensitivity.pdf
│   ├── fig_*.png               # 预测模型图
│   └── *.tex                   # LaTeX表格
├── python作业.tex              # 最终报告（LaTeX）
└── portfolio_files/            # GitHub portfolio文件
```

---

## 技能展示

| 类别 | 技能 |
|------|------|
| **因果推断** | 断点回归设计、局部多项式回归、操纵性检验 |
| **数据分析** | 大规模纵向数据处理、缺失值处理、特征工程 |
| **统计方法** | 加权最小二乘、核函数加权、协变量均衡检验 |
| **机器学习** | 线性/岭回归、随机森林、梯度提升、模型评估 |
| **可视化** | 学术出版级图表、雷达图、特征重要性图 |
| **学术写作** | LaTeX文档排版、XeLaTeX中文支持 |

---

*本项目为北京大学定量研究方法课程作业（2024-2025学年）*

**联系方式**: 2511110239@stu.pku.edu.cn
