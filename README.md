# 新能源发电预测模型优化

## 项目概述

本项目针对新能源发电预测任务进行了一系列代码优化和模型改进。通过对气象数据和发电功率数据的深入分析与处理，构建了一个高精度的发电功率预测系统。项目主要包括数据处理、特征工程、模型训练与融合等环节，每个环节都进行了针对性的优化，以提高预测精度。

## 数据说明

项目使用的数据包括：

- **气象数据**：来自初赛训练集中的NWP数据，包含ghi（全球水平辐照度）、poai（平面辐照度）、sp（表面气压）、t2m（2米温度）、tcc（总云量）、tp（总降水量）、u100（100米U风速）、v100（100米V风速）等8个气象要素。
- **发电功率数据**：来自初赛训练集中的功率数据，记录了每15分钟的发电功率值。

## 代码优化亮点

### 1. 增强数据提取

原始代码仅提取了气象数据的均值特征，我对此进行了扩展，增加了多种统计特征的提取：

```python
# 提取基础特征 - 均值
mean_values = np.array([np.mean(data[:, :, i, :, :][0], axis=(1, 2)) for i in range(8)]).T
mean_df = pd.DataFrame(mean_values, columns=channel)

# 提取最大值
max_values = np.array([np.max(data[:, :, i, :, :][0], axis=(1, 2)) for i in range(8)]).T
max_df = pd.DataFrame(max_values, columns=[f"{col}_max" for col in channel])

# 提取最小值、标准差、中位数、四分位距等
```

这种多维度的特征提取能够更全面地捕捉气象数据的分布特性，为模型提供更丰富的信息。

### 2. 数据可视化与分析

增加了相关性分析和特征重要性分析，帮助理解特征与目标变量的关系：

```python
# 相关性分析
base_features = ['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100', 'power']
corr_matrix = train_data[base_features].corr()
plt.figure(figsize=(12, 10))
plt.title('特征相关性热力图')
sns_plot = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
```

通过可视化分析，可以直观地发现哪些特征对发电功率影响更大，为后续特征工程提供指导。

### 3. 增强数据清洗

实现了更完善的数据清洗流程，包括缺失值处理、异常值检测与处理：

```python
# 异常值处理
numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'power':  # 不处理目标变量
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 将异常值替换为边界值
        df_copy.loc[df_copy[col] < lower_bound, col] = lower_bound
        df_copy.loc[df_copy[col] > upper_bound, col] = upper_bound
```

这种处理方式可以有效减少异常值对模型训练的干扰，提高模型的稳定性。

### 4. 增强特征工程

大幅扩展了特征工程部分，添加了多种类型的特征：

- **基础特征**：如风速特征
- **时间特征**：小时特征、周期性时间特征
- **交互特征**：温度与风速的交互、太阳辐射与温度的交互等
- **滞后特征**：前一个小时的特征值
- **差分特征**：当前值与前一个小时的差值
- **滑动窗口统计特征**：过去3小时的平均值、标准差等
- **多项式特征**：对重要特征进行二次项变换

```python
# 风速特征
df_copy["wind_speed"] = np.sqrt(df_copy['u100']**2 + df_copy['v100']**2)

# 时间特征
df_copy["hour"] = df_copy.index % 24
df_copy["hour_sin"] = np.sin(2 * np.pi * df_copy["hour"] / 24)
df_copy["hour_cos"] = np.cos(2 * np.pi * df_copy["hour"] / 24)
```

这些丰富的特征能够帮助模型更好地理解数据中的模式和规律。

### 5. 多模型融合

实现了LightGBM、XGBoost和CatBoost三种模型的融合，通过加权平均的方式结合各模型的预测结果：

```python
# 计算每个模型的权重（基于验证集得分）
total_score = sum(models_scores.values())
weights = {model: score/total_score for model, score in models_scores.items()}

# 加权融合预测结果
ensemble_preds = np.zeros(test_x.shape[0])
for model, preds in models_preds.items():
    ensemble_preds += weights[model] * preds
```

模型融合可以有效降低单一模型的方差，提高预测的稳定性和准确性。

### 6. 参数优化

使用网格搜索寻找最佳模型参数，提高模型性能：

```python
# 定义参数网格
param_grid = {
    'num_leaves': [31, 63, 127],
    'learning_rate': [0.05, 0.1, 0.2],
    'min_child_weight': [3, 5, 7],
    'feature_fraction': [0.7, 0.8, 0.9],
    'bagging_fraction': [0.7, 0.8, 0.9],
    'lambda_l2': [5, 10, 15]
}

# 使用网格搜索找到最佳参数
```

通过系统化的参数搜索，可以找到更适合当前数据集的模型配置，进一步提升预测性能。

## 性能评估

项目使用交叉验证评估模型性能，采用RMSE（均方根误差）作为评价指标，并通过1/(1+RMSE)转换为得分。通过多模型融合和参数优化，最终模型在验证集上取得了显著的性能提升。

## 总结与展望

本项目通过一系列的代码优化和模型改进，成功提高了新能源发电预测的准确性。主要贡献包括：

1. 多维度特征提取，全面捕捉气象数据特性
2. 完善的数据清洗流程，提高数据质量
3. 丰富的特征工程，增强模型表达能力
4. 多模型融合策略，提升预测稳定性和准确性
5. 参数优化，找到最适合数据的模型配置

未来可以进一步探索深度学习模型在时序预测中的应用，以及考虑更多外部因素（如电网负荷、历史发电模式等）对预测的影响。

## 环境依赖

```
python>=3.7
numpy
pandas
matplotlib
scikit-learn
lightgbm
xgboost
catboost
netCDF4
tqdm
```

---

*本项目代码由本人独立完成，仅用于学习交流。*
