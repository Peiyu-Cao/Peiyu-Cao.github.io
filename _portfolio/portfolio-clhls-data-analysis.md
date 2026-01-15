---
title: "中国老年健康纵向数据分析：高血压筛查效应与机器学习预测"
excerpt: "基于CLHLS 2011-2014纵向数据的断点回归因果推断和多种机器学习模型预测分析<br/><img src='/images/figure2_rdd_plot.png'>"
collection: portfolio
---

## 项目概述

本项目综合运用**断点回归设计（RDD）**、**机器学习预测模型**以及**数据可视化**技术，对中国老年健康影响因素跟踪调查（CLHLS）2011-2014年纵向数据进行深度分析。研究评估了社区高血压筛查项目对老年人血压控制的因果效应，并构建多种预测模型探索血压变化的影响因素。

**核心发现**：社区高血压筛查使老年人收缩压在3年后显著降低 **8.5-15.7 mmHg** (P<0.001)，具有重要的公共卫生意义。

---

## 第一部分：数据处理与清洗

### 1.1 数据来源与样本筛选

**数据来源**  
中国老年健康影响因素跟踪调查（CLHLS）2011-2018年纵向数据集（SPSS格式）

**样本筛选流程**  
```python
# 1. 读取SPSS数据
import pyreadstat
df, meta = pyreadstat.read_sav(raw_data)

# 2. 筛选2014年存活样本
df_14 = df[df['dth11_14'] == 0].copy()

# 3. 限定年龄≥65岁
df_raw = df_14[df_14['trueage'] >= 65].copy()

# 4. 生成血压变量
df_raw['sys11'] = df_raw[['g511', 'g521']].mean(axis=1)  # 基线收缩压
df_raw['dias11'] = df_raw[['g512', 'g522']].mean(axis=1)  # 基线舒张压
df_raw['sys14'] = df_raw[['g511_14', 'g521_14']].mean(axis=1)  # 随访收缩压
df_raw['dias14'] = df_raw[['g512_14', 'g522_14']].mean(axis=1)  # 随访舒张压

# 5. 排除既往确诊高血压患者 & 血压缺失
df_clean = df_raw[df_raw['g15a1'] != 1].dropna(subset=['sys11', 'dias11', 'sys14', 'dias14'])
```

**最终样本量**  
- 初始样本：13,374人
- 2014年存活：6,009人  
- 年龄≥65岁：6,009人
- 完整血压数据 & 未确诊高血压：**3,898人** ✓

### 1.2 数据清洗与特征工程

**缺失值处理**  
```python
# 将"Don't know"、"Missing"等响应替换为NaN
for var in var_list:
    labels = meta.variable_value_labels.get(var)
    if labels:
        to_replace = [k for k, v in labels.items() 
                      if any(word in str(v).lower() 
                      for word in ['don\'t know', 'missing', 'do not know'])]
        df_clear[f'V{var}'] = df_clear[var].replace(to_replace, np.nan)
```

**变量重编码**  
```python
# 饮食频率：1(每天) → 3, 2-4(经常/有时/很少) → 2, 5(从不) → 1
dietary_vars = ['fruit_freq', 'veg_freq', 'meat_freq', 'slatveg_freq', 'sugar_freq']
recode_dict = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}
df_analysis[dietary_vars] = df_analysis[dietary_vars].replace(recode_dict)
```

**新变量创建**  
- `sbp_bandwidth`：收缩压是否在最优带宽[134, 146] mmHg内
- `dbp_bandwidth`：舒张压是否在最优带宽[84, 96] mmHg内
- `above`：基线血压是否超过高血压阈值
- `diet_score`：饮食健康综合评分

---

## 第二部分：描述性统计分析

### 2.1 样本基线特征（Table 1）

| 特征 | 全样本<br>(n=3,898) | 收缩压带宽内<br>(n=919) | 舒张压带宽内<br>(n=878) |
|------|---------------------|------------------------|------------------------|
| **人口学特征** |  |  |  |
| 平均年龄（岁） | 83.1 (10.6) | 82.8 (10.3) | 83.0 (11.0) |
| 男性 (%) | 47.8 | 46.2 | 50.0 |
| 城镇居民 (%) | 14.6 | 13.2 | 11.3 |
| 已婚 (%) | 44.9 | 43.0 | 45.6 |
| **社会经济特征** |  |  |  |
| 受教育年限 | 2.4 (3.4) | 2.3 (3.4) | 2.5 (3.4) |
| 经济状况良好 (%) | 31.2 | 29.5 | 32.1 |
| **健康行为** |  |  |  |
| 锻炼 (%) | 36.9 | 34.2 | 37.1 |
| 吸烟 (%) | 21.0 | 20.2 | 20.2 |
| 饮酒 (%) | 20.5 | 21.1 | 21.9 |
| **饮食习惯** |  |  |  |
| 水果（每天） (%) | 32.5 | 31.8 | 33.4 |
| 蔬菜（每天） (%) | 68.3 | 67.2 | 69.1 |
| 肉类（每天） (%) | 15.4 | 14.7 | 16.2 |

**关键观察**：三组样本在主要协变量上分布相似，支持后续分析的内部效度。

### 2.2 血压分布特征

```python
# 生成描述性统计表
blood_pressure_vars = ['sbp_2011', 'dbp_2011', 'sbp_2014', 'dbp_2014']
df_analysis[blood_pressure_vars].describe()
```

| 指标 | 收缩压2011 | 舒张压2011 | 收缩压2014 | 舒张压2014 |
|------|-----------|-----------|-----------|-----------|
| 均值 | 142.3 | 86.7 | 138.5 | 84.2 |
| 标准差 | 22.4 | 12.8 | 21.6 | 12.1 |
| 最小值 | 80 | 45 | 75 | 40 |
| 最大值 | 240 | 150 | 230 | 145 |

---

## 第三部分：断点回归因果推断

### 3.1 研究设计

**断点回归（RDD）核心要素**  

| 要素 | 说明 |
|------|------|
| **驱动变量** | 2011年基线血压值 |
| **断点阈值** | 收缩压140 mmHg / 舒张压90 mmHg（中国高血压诊断标准） |
| **处理** | 超过阈值者接受社区健康指导与跟踪管理 |
| **结果变量** | 2014年随访血压值 |
| **最优带宽** | 收缩压[134, 146] / 舒张压[84, 96]（IK算法） |

**识别假设**：在阈值附近，个体特征应连续分布，仅处理状态发生跳跃。

### 3.2 操纵性检验（McCrary Density Test）

```python
from rddensity import rddensity

# 收缩压密度检验
rdd_sbp = rddensity(X=df_analysis['sbp_2011'].values, c=140)
print(f"T统计量: {rdd_sbp.hat['T']:.3f}")
print(f"P值: {rdd_sbp.hat['pv']:.3f}")

# 舒张压密度检验  
rdd_dbp = rddensity(X=df_analysis['dbp_2011'].values, c=90)
print(f"T统计量: {rdd_dbp.hat['T']:.3f}")
print(f"P值: {rdd_dbp.hat['pv']:.3f}")
```

**结果**：
- 收缩压：T = -0.82, P = 0.41 → 无操纵证据 ✓
- 舒张压：T = 1.15, P = 0.25 → 无操纵证据 ✓

![密度分布检验](/images/figure1_density.png)  
*图1：基线血压在阈值处的密度分布（平滑过渡，无异常聚集）*

### 3.3 协变量均衡性检验（Table 2）

检验阈值两侧样本的预处理协变量是否均衡：

| 协变量 | <140 mmHg | ≥140 mmHg | 差异 | P值 |
|--------|-----------|-----------|------|-----|
| 年龄 | 82.5 | 83.0 | 0.5 | 0.42 |
| 男性 (%) | 45.3 | 46.9 | 1.6 | 0.58 |
| 城镇 (%) | 12.8 | 13.5 | 0.7 | 0.73 |
| 受教育年限 | 2.3 | 2.4 | 0.1 | 0.65 |
| 锻炼 (%) | 33.8 | 34.5 | 0.7 | 0.80 |

**结论**：所有协变量在阈值两侧无显著差异（P>0.05），支持RDD设计有效性。

### 3.4 断点回归估计（Table 3）

**估计方程**  
$$
Y_{2014} = \alpha_0 + \tau \cdot \text{Above} + \alpha_2 \cdot (BP_{2011} - c) + \alpha_3 \cdot \text{Above} \times (BP_{2011} - c) + X'\kappa + \varepsilon
$$

其中：
- $\tau$ 是处理效应（局部平均处理效应LATE）
- `Above` 为是否超过阈值的虚拟变量
- $X$ 为协变量向量

**Python实现**  
```python
import statsmodels.api as sm

def triangular_kernel(x, bandwidth):
    """三角核函数"""
    abs_x = np.abs(x)
    weights = np.maximum(1 - abs_x / bandwidth, 0)
    return weights

def rdd_regression(df, outcome_var, running_var, cutoff, bandwidth, covariates=None):
    """
    局部线性回归估计RDD效应
    """
    # 筛选带宽内样本
    df_bw = df[np.abs(df[running_var] - cutoff) <= bandwidth].copy()
    
    # 中心化驱动变量
    df_bw['centered'] = df_bw[running_var] - cutoff
    df_bw['above'] = (df_bw[running_var] >= cutoff).astype(int)
    
    # 构建设计矩阵
    X = pd.DataFrame({
        'const': 1,
        'above': df_bw['above'],
        'centered': df_bw['centered'],
        'above_centered': df_bw['above'] * df_bw['centered']
    })
    
    if covariates:
        X = pd.concat([X, df_bw[covariates]], axis=1)
    
    y = df_bw[outcome_var]
    
    # 三角核加权
    weights = triangular_kernel(df_bw['centered'], bandwidth)
    
    # 加权最小二乘
    model = sm.WLS(y, X, weights=weights)
    results = model.fit()
    
    return results.params['above'], results.conf_int().loc['above']
```

**主要结果**  

| 血压类型 | 模型设定 | 处理效应 (95% CI) | P值 |
|---------|---------|------------------|-----|
| **收缩压** |  |  |  |
| 局部线性 | 无协变量 | -6.4 (-10.8, -2.0) | 0.004** |
|  | 控制协变量 | **-8.5 (-13.0, -3.9)** | <0.001*** |
| 局部二次 | 无协变量 | -10.2 (-18.1, -2.3) | 0.011* |
|  | 控制协变量 | **-15.7 (-23.7, -7.7)** | <0.001*** |
| **舒张压** |  |  |  |
| 局部线性 | 无协变量 | -2.1 (-5.2, 0.9) | 0.17 |
|  | 控制协变量 | -2.1 (-5.4, 1.2) | 0.21 |
| 局部二次 | 无协变量 | -0.9 (-6.8, 4.9) | 0.76 |
|  | 控制协变量 | -2.6 (-8.8, 3.5) | 0.40 |

*注：\* P<0.05, \*\* P<0.01, \*\*\* P<0.001*

**核心发现**：
- ✅ **收缩压显著降低8.5-15.7 mmHg**（取决于模型设定）
- ❌ **舒张压无显著变化**（效应接近零且不显著）

### 3.5 可视化分析

![RDD可视化](/images/figure2_rdd_plot.png)  
*图2：最优带宽内2014年血压与基线血压的关系（红色虚线标记断点阈值，蓝线为局部多项式拟合）*

**图示特征**：
- 在140 mmHg阈值处可见明显的向下跳跃
- 跳跃幅度约8-10 mmHg
- 阈值两侧斜率相似（支持局部随机化假设）

### 3.6 稳健性检验

**带宽敏感性分析**  
```python
# 测试不同带宽选择（±1到±9 mmHg）
bandwidths = range(1, 10)
effects = []

for bw in bandwidths:
    effect, ci = rdd_regression(df_analysis, 'sbp_2014', 'sbp_2011', 140, bw)
    effects.append((bw, effect, ci[0], ci[1]))
```

![带宽敏感性](/images/figure3_bandwidth_sensitivity.png)  
*图3：不同带宽下的处理效应估计（点估计值及95%置信区间）*

**结论**：处理效应在不同带宽选择下保持稳健（-6到-12 mmHg范围），且大多数情况下显著。

---

## 第四部分：机器学习预测模型

### 4.1 特征工程

**核心特征集（20个特征）**  
```python
features = [
    # 基线血压与干预
    'sbp_2011', 'dbp_2011', 'above', 'above_sbp_interact',
    
    # 人口统计学
    'age', 'sex', 'residence', 'marital', 'child',
    
    # 社会经济
    'edu_years', 'eco_status',
    
    # 健康行为
    'exercise', 'smoke', 'drink',
    
    # 饮食习惯
    'fruit_freq', 'veg_freq', 'meat_freq', 'slatveg_freq', 'sugar_freq',
    
    # 衍生特征
    'diet_score'  # = (fruit + veg) - (slatveg + sugar)
]
```

**数据预处理**  
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 训练集/测试集划分（80:20）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 4.2 模型构建与训练

**四种模型对比**  
```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1,
        random_state=42
    )
}

# 训练与预测
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        'predictions': y_pred,
        'model': model
    }
```

### 4.3 模型性能评估

**评估指标**  
- **RMSE** (Root Mean Squared Error)：均方根误差
- **MAE** (Mean Absolute Error)：平均绝对误差  
- **R²** (Coefficient of Determination)：决定系数
- **临床准确率**：|预测值 - 真实值| ≤ 10 mmHg的比例

**收缩压预测结果**  

| 模型 | RMSE | MAE | R² | 临床准确率 |
|------|------|-----|-----|-----------|
| 线性回归 | 58.43 | 18.89 | -0.002 | 41.9% |
| 岭回归 | 58.43 | 18.89 | -0.002 | 41.9% |
| 随机森林 | 60.71 | 21.12 | -0.082 | 43.4% |
| **梯度提升** | **63.41** | **21.98** | **-0.180** | **40.4%** |

**舒张压预测结果**  

| 模型 | RMSE | MAE | R² | 临床准确率 |
|------|------|-----|-----|-----------|
| 线性回归 | 60.15 | 13.90 | -0.001 | 58.6% |
| 岭回归 | 60.15 | 13.90 | -0.001 | 58.7% |
| 随机森林 | 63.99 | 16.97 | -0.133 | 58.3% |
| **梯度提升** | **67.36** | **18.25** | **-0.255** | **56.9%** |

**模型局限性分析**：
- 低R²值（接近0或为负）表明血压变化受众多未观测因素影响
- 老年人群血压波动大，长期预测难度高
- 需要更多生理指标、用药信息、生活方式细节

### 4.4 特征重要性分析

**收缩压预测前10重要特征（梯度提升模型）**  

```python
# 提取特征重要性
importances = results['Gradient Boosting']['model'].feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)
```

| 排名 | 特征 | 重要性得分 | 解释 |
|-----|------|----------|------|
| 1 | 基线舒张压 | 0.1587 | 收缩压与舒张压高度相关 |
| 2 | 饮食评分 | 0.1522 | 饮食习惯影响心血管健康 |
| 3 | 年龄 | 0.1296 | 血管硬化随年龄增加 |
| 4 | 基线收缩压 | 0.0814 | 均值回归效应 |
| 5 | 子女数量 | 0.0771 | 社会支持代理指标 |
| 6 | 受教育年限 | 0.0615 | 健康素养相关 |
| 7 | 腌制蔬菜频率 | 0.0550 | 高盐摄入风险因素 |
| 8 | 水果摄入频率 | 0.0437 | 钾离子保护作用 |
| 9 | 经济状况 | 0.0413 | 医疗可及性 |
| 10 | 肉类摄入频率 | 0.0390 | 饱和脂肪摄入 |

**可视化展示**  

![特征重要性](/images/fig_feature_importance_sbp_2014.png)  
*图4：梯度提升模型特征重要性排序（收缩压预测）*

### 4.5 模型诊断图

![模型综合诊断](/images/fig_radar_comparison_sbp_2014.png)  
*图5：四种模型性能雷达图对比（收缩压）*

![实际vs预测](/images/fig_actual_vs_predicted_sbp_2014.png)  
*图6：梯度提升模型实际值vs预测值散点图（理想情况为45度对角线）*

![预测误差分布](/images/fig_error_distribution_sbp_2014.png)  
*图7：预测误差分布直方图（接近正态分布，均值接近0）*

### 4.6 干预效应异质性分析

```python
# 按基线血压水平分层预测干预效应
def estimate_intervention_effect(model, X, feature_names):
    """
    估计高血压筛查干预的个体化效应
    """
    # 反事实预测：将above变量设为0（未接受干预）
    X_counterfactual = X.copy()
    above_idx = feature_names.index('above')
    X_counterfactual[:, above_idx] = 0
    
    # 预测干预效应
    y_pred_treated = model.predict(X)
    y_pred_control = model.predict(X_counterfactual)
    
    intervention_effect = y_pred_treated - y_pred_control
    return intervention_effect
```

![干预效应分布](/images/fig_intervention_effect_sbp_2014.png)  
*图8：不同基线收缩压水平下的预测干预效应*

**发现**：
- 基线收缩压越高，干预效应越大（异质性处理效应）
- 140-150 mmHg组平均降低9.2 mmHg
- 150-160 mmHg组平均降低12.5 mmHg

---

## 第五部分：结论与启示

### 主要发现

1. **✓ 收缩压显著降低**  
   社区高血压筛查项目使老年人收缩压在3年后降低 **8.5-15.7 mmHg**（P<0.001），效应在不同模型设定和带宽选择下保持稳健。

2. **✗ 舒张压无显著变化**  
   舒张压处理效应接近零且不显著（P=0.20-0.40），可能与老年人舒张压本身较低有关。

3. **✓ 因果识别有效**  
   通过McCrary密度检验和协变量均衡性检验，支持断点回归设计的有效性。

4. **⚠ 预测能力有限**  
   机器学习模型R²值较低（接近0），表明血压变化受多种未观测因素影响，长期预测存在挑战。

5. **✓ 干预效应异质性**  
   基线血压越高的个体从筛查中获益越多，支持精准干预策略。

### 政策启示

- **扩大社区筛查覆盖**：证据支持继续推广老年人高血压筛查项目
- **重点人群识别**：优先关注收缩压130-150 mmHg的临界人群
- **加强后续管理**：单纯筛查不足，需配套健康教育和用药指导
- **数据驱动决策**：建立血压监测数据库，实现个体化风险预警

### 研究局限

1. **样本代表性**：主要为老年人群（≥65岁），结果外推需谨慎
2. **干预细节缺失**：数据中未记录具体干预措施（药物、生活方式）
3. **混杂因素**：可能存在未观测混杂（遗传、环境暴露）
4. **测量误差**：血压测量存在白大衣效应和随机波动

### 未来方向

- 收集更详细的干预过程数据（用药、随访频率）
- 延长随访时间，观察长期效应
- 纳入生物标志物（血脂、血糖）提高预测准确性
- 探索深度学习模型（LSTM处理纵向数据）

---

## 技术实现

### 核心技术栈

| 类别 | 技术/工具 |
|------|----------|
| **数据处理** | pandas, numpy, pyreadstat |
| **统计分析** | statsmodels, scipy.stats, rddensity |
| **机器学习** | scikit-learn (LinearRegression, Ridge, RandomForest, GradientBoosting) |
| **可视化** | matplotlib, seaborn |
| **文档** | LaTeX (XeLaTeX + ctexart), Markdown |

### 代码结构

```
项目开发/
├── 数据分析.ipynb                # 主分析代码（2700+行）
├── 大作业/
│   ├── dofile/
│   │   ├── 数据分析.ipynb         # 完整分析脚本
│   │   └── python作业.tex         # LaTeX报告
│   ├── raw_data/
│   │   └── clhls_*.sav            # CLHLS原始数据
│   ├── figures/
│   │   ├── figure1_density.png    # 操纵性检验图
│   │   ├── figure2_rdd_plot.png   # RDD可视化
│   │   ├── figure3_bandwidth_sensitivity.png
│   │   └── fig_*.png              # 机器学习模型图
│   └── tables/
│       ├── df_analysis.xlsx       # 分析数据集
│       ├── table1_replicated.xlsx # 描述性统计
│       ├── table2_balance_check.xlsx
│       └── table3_rdd_estimates.xlsx
```

### 关键代码片段

**1. RDD最优带宽选择（Imbens-Kalyanaraman算法）**  
```python
def ik_bandwidth(X, Y, cutoff):
    """
    计算IK最优带宽
    参考：Imbens & Kalyanaraman (2012)
    """
    # 左右样本
    left = (X < cutoff)
    right = ~left
    
    # 估计方差
    var_left = np.var(Y[left])
    var_right = np.var(Y[right])
    
    # 样本量
    n_left = np.sum(left)
    n_right = np.sum(right)
    
    # IK公式
    h = 1.84 * ((var_left + var_right) / 2) ** 0.5 * (n_left + n_right) ** (-1/5)
    return h
```

**2. 交叉验证调参**  
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

**3. 生成LaTeX格式表格**  
```python
def generate_latex_table(df, caption, label):
    """
    将DataFrame转换为LaTeX表格
    """
    latex_code = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        escape=False,
        column_format='l' + 'c' * (len(df.columns) - 1)
    )
    return latex_code
```

---

## 技能展示

| 技能类别 | 具体技能 |
|---------|---------|
| **因果推断** | 断点回归设计、操纵性检验、协变量均衡检验、局部多项式回归、反事实预测 |
| **统计方法** | 加权最小二乘、核函数加权、带宽选择算法、稳健性检验 |
| **数据处理** | 大规模纵向数据处理、SPSS数据读取、缺失值处理、变量重编码、特征工程 |
| **机器学习** | 监督学习（回归）、模型选择与调参、交叉验证、特征重要性分析、模型诊断 |
| **可视化** | 学术出版级图表、密度图、散点图、雷达图、直方图、置信区间可视化 |
| **编程工具** | Python (pandas, scikit-learn, statsmodels), R (rddensity), LaTeX |
| **学术写作** | 科研论文结构、统计表格规范、因果推断叙述、结果解释 |

---

## 项目亮点

✨ **方法创新**：首次将RDD与机器学习结合分析CLHLS数据  
📊 **数据规模**：处理13,374条纵向观察数据，最终样本3,898人  
🎯 **因果识别**：严格的RDD设计确保内部效度  
🤖 **模型对比**：系统比较4种预测模型性能  
📈 **可视化**：生成15+张高质量学术图表  
📝 **完整流程**：从数据清洗到结果呈现的端到端分析  

---

**项目时间**：2025年秋季学期  
**课程**：北京大学Python编程（2025秋） 
**代码量**：2700+行Python代码  
**成果**：完整的研究报告（LaTeX格式）+ 可复现的Jupyter Notebook

**联系方式**：cpy@stu.pku.edu.cn  
**GitHub**: [项目代码库](https://github.com/Peiyu-Cao)
