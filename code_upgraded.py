from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 1.查看训练集里的气象数据
nc_path = "data/初赛训练集/nwp_data_train/1/NWP_1/20240101.nc"
dataset = Dataset(nc_path, mode='r')
dataset.variables.keys() # 查看数据的变量
# 输出：dict_keys(['time', 'channel', 'data', 'lat', 'lon', 'lead_time'])

channel = dataset.variables["channel"][:] #
channel 
# 输出：array(['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100'], dtype=object)

data = dataset.variables["data"][:]
data.shape
# 输出：(1, 24, 8, 11, 11)

# 2.数据处理
'''观测上步代码结果可知气象数据中
新能源场站每个小时的数据维度为2（11x11），一个矩阵
但构建模型对于单个时间点的单个特征只需要一个 标量 即可，
因此我们把 11x11 个格点的数据取均值，
从而将二维数据转为单一的标量值。
且主办方提供的气象数据时间精度为h，
而发电功率精度为15min，
即给我们一天的数据有24条天气数据与96（24*4）条功率数据，
因此将功率数据中每四条数据只保留一条。
'''
date_range = pd.date_range(start='2024-01-01', end='2024-12-30')
# 将%Y-%m-%d格式转为%Y%m%d
date = [date.strftime('%Y%m%d') for date in date_range]

# 定义读取训练/测试集函数
# 优化点1：增强数据提取，不仅提取均值，还提取最大值、最小值、标准差等统计特征
def get_data(path_template, date):
    # 读取该天数据
    dataset = Dataset(path_template.format(date), mode='r') #read-only
    # 获取列名
    channel = dataset.variables["channel"][:]
    # 获取列名对应的数据
    data = dataset.variables["data"][:]
    
    # 创建一个空的DataFrame用于存储所有特征
    all_features = []
    
    # 提取基础特征 - 均值
    mean_values = np.array([np.mean(data[:, :, i, :, :][0], axis=(1, 2)) for i in range(8)]).T
    mean_df = pd.DataFrame(mean_values, columns=channel)
    
    # 优化点2：提取更多统计特征
    # 提取最大值
    max_values = np.array([np.max(data[:, :, i, :, :][0], axis=(1, 2)) for i in range(8)]).T
    max_df = pd.DataFrame(max_values, columns=[f"{col}_max" for col in channel])
    
    # 提取最小值
    min_values = np.array([np.min(data[:, :, i, :, :][0], axis=(1, 2)) for i in range(8)]).T
    min_df = pd.DataFrame(min_values, columns=[f"{col}_min" for col in channel])
    
    # 提取标准差 - 反映数据的波动性
    std_values = np.array([np.std(data[:, :, i, :, :][0], axis=(1, 2)) for i in range(8)]).T
    std_df = pd.DataFrame(std_values, columns=[f"{col}_std" for col in channel])
    
    # 提取中位数 - 对异常值不敏感
    median_values = np.array([np.median(data[:, :, i, :, :][0], axis=(1, 2)) for i in range(8)]).T
    median_df = pd.DataFrame(median_values, columns=[f"{col}_median" for col in channel])
    
    # 提取四分位距 - 反映数据的分散程度
    q75_values = np.array([np.percentile(data[:, :, i, :, :][0], 75, axis=(1, 2)) for i in range(8)]).T
    q25_values = np.array([np.percentile(data[:, :, i, :, :][0], 25, axis=(1, 2)) for i in range(8)]).T
    iqr_values = q75_values - q25_values
    iqr_df = pd.DataFrame(iqr_values, columns=[f"{col}_iqr" for col in channel])
    
    # 合并所有特征
    result_df = pd.concat([mean_df, max_df, min_df, std_df, median_df, iqr_df], axis=1)
    
    return result_df

# 定义路径模版：{}.nc 表示占位符 会在format中被替换为具体的日期
train_path_template = "/sdc/model/data/初赛训练集/nwp_data_train/1/NWP_1/{}.nc"
# 通过列表推导式获取数据 返回的列表中每个元素都是以天为单位的数据
data = [get_data(train_path_template, i) for i in tqdm(date, desc="加载训练数据")]
# 将每天的数据拼接并重设index
train = pd.concat(data, axis=0).reset_index(drop=True)
# 读取目标值
target = pd.read_csv("/sdc/model/data/初赛训练集/fact_data/1_normalization_train.csv")
target = target[96:]
# 功率数据中每四条数据去掉三条
target = target[target['时间'].str.endswith('00:00')]
target = target.reset_index(drop=True)
# 将目标值合并到训练集
train["power"] = target["功率(MW)"]

# 3.数据可视化与分析
# 优化点3：增加相关性分析，帮助理解特征与目标变量的关系
def visualize_data(train_data):
    # 基础可视化
    hours = range(24)
    plt.figure(figsize=(20,10))
    # 绘制八个基础特征及目标值
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.plot(hours, train_data.iloc[:24, i])
        plt.title(train_data.columns.tolist()[i])
    plt.tight_layout()
    plt.savefig('feature_visualization.png')
    plt.close()
    
    # 相关性分析
    # 只选择基础特征进行相关性分析，避免图表过于复杂
    base_features = ['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100', 'power']
    corr_matrix = train_data[base_features].corr()
    plt.figure(figsize=(12, 10))
    plt.title('特征相关性热力图')
    sns_plot = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    # 特征重要性分析（使用简单的随机森林模型）
    X = train_data.drop('power', axis=1)
    y = train_data['power']
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 获取特征重要性
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 只展示前20个重要特征
    top_n = 20
    plt.figure(figsize=(12, 8))
    plt.title('特征重要性排名')
    plt.bar(range(top_n), importances[indices[:top_n]], align='center')
    plt.xticks(range(top_n), X.columns[indices[:top_n]], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("数据可视化完成，图表已保存")

# 4.数据清洗
# 优化点4：增强数据清洗，包括缺失值处理、异常值处理和特征标准化
def clean_data(df):
    print("开始数据清洗...")
    # 复制一份数据
    df_copy = df.copy()
    
    # 1. 缺失值处理
    # 统计缺失值
    missing_values = df_copy.isnull().sum()
    print(f"缺失值统计:\n{missing_values[missing_values > 0]}")
    
    # 对于缺失比例小于5%的特征，使用中位数填充
    for col in df_copy.columns:
        missing_ratio = df_copy[col].isnull().mean()
        if 0 < missing_ratio < 0.05:
            print(f"使用中位数填充特征 {col} 的缺失值")
            df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    # 2. 异常值处理
    # 使用IQR方法检测并处理异常值
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'power':  # 不处理目标变量
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 统计异常值数量
            outliers = ((df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"特征 {col} 有 {outliers} 个异常值")
                
                # 将异常值替换为边界值
                df_copy.loc[df_copy[col] < lower_bound, col] = lower_bound
                df_copy.loc[df_copy[col] > upper_bound, col] = upper_bound
    
    # 3. 删除剩余的含有缺失值的行
    rows_before = df_copy.shape[0]
    df_copy = df_copy.dropna().reset_index(drop=True)
    rows_after = df_copy.shape[0]
    print(f"删除了 {rows_before - rows_after} 行含有缺失值的数据")
    
    return df_copy

# 5.特征工程
# 优化点5：增强特征工程，添加时间特征、交互特征和统计特征
def feature_engineering(df):
    print("开始特征工程...")
    # 复制一份数据
    df_copy = df.copy()
    
    # 1. 基础特征
    # 风速特征
    df_copy["wind_speed"] = np.sqrt(df_copy['u100']**2 + df_copy['v100']**2)
    
    # 2. 时间特征
    # 小时特征
    df_copy["hour"] = df_copy.index % 24
    
    # 添加周期性时间特征（将小时转换为周期性特征）
    df_copy["hour_sin"] = np.sin(2 * np.pi * df_copy["hour"] / 24)
    df_copy["hour_cos"] = np.cos(2 * np.pi * df_copy["hour"] / 24)
    
    # 添加日期特征
    # 假设数据是按时间顺序排列的，每天24个小时
    df_copy["day"] = df_copy.index // 24
    
    # 3. 交互特征
    # 温度与风速的交互
    df_copy["temp_wind"] = df_copy["t2m"] * df_copy["wind_speed"]
    
    # 太阳辐射与温度的交互
    df_copy["ghi_temp"] = df_copy["ghi"] * df_copy["t2m"]
    
    # 太阳辐射与云量的交互
    df_copy["ghi_tcc"] = df_copy["ghi"] * df_copy["tcc"]
    
    # 4. 滞后特征（前一个小时的特征）
    for col in ['ghi', 'poai', 't2m', 'wind_speed']:
        df_copy[f"{col}_lag1"] = df_copy[col].shift(1)
        df_copy[f"{col}_lag2"] = df_copy[col].shift(2)
    
    # 5. 差分特征（当前值与前一个小时的差值）
    for col in ['ghi', 'poai', 't2m', 'wind_speed']:
        df_copy[f"{col}_diff1"] = df_copy[col].diff(1)
    
    # 6. 滑动窗口统计特征
    for col in ['ghi', 'poai', 't2m', 'wind_speed']:
        # 过去3小时的平均值
        df_copy[f"{col}_rolling_mean_3"] = df_copy[col].rolling(window=3, min_periods=1).mean()
        # 过去3小时的标准差
        df_copy[f"{col}_rolling_std_3"] = df_copy[col].rolling(window=3, min_periods=1).std()
    
    # 7. 多项式特征
    # 对重要特征进行二次项变换
    for col in ['ghi', 'poai', 't2m']:
        df_copy[f"{col}_squared"] = df_copy[col] ** 2
    
    # 8. 特征标准化
    # 对数值型特征进行标准化处理
    # 注意：不对目标变量进行标准化
    scaler = StandardScaler()
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    if 'power' in numeric_cols:
        numeric_cols.remove('power')
    
    # 填充缺失值
    for col in numeric_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].median())
    
    # 应用标准化
    df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
    
    print(f"特征工程完成，特征数量从 {df.shape[1]} 增加到 {df_copy.shape[1]}")
    return df_copy

# 6.模型训练与验证
# 优化点6：多模型融合，结合LightGBM、XGBoost和CatBoost的预测结果
def train_models(train_data, test_data):
    print("开始模型训练...")
    # 获取特征和目标变量
    X = train_data.drop('power', axis=1)
    y = train_data['power']
    
    # 获取特征列名
    feature_cols = X.columns.tolist()
    
    # 准备测试集
    test_x = test_data[feature_cols]
    
    # 定义交叉验证
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True, random_state=2024)
    
    # 存储每个模型的验证结果和测试集预测结果
    models_oof = {}
    models_preds = {}
    models_scores = {}
    
    # 1. LightGBM模型
    print("训练LightGBM模型...")
    lgb_oof, lgb_preds, lgb_scores = cv_model_lgb(X, y, test_x, kf)
    models_oof['lgb'] = lgb_oof
    models_preds['lgb'] = lgb_preds
    models_scores['lgb'] = np.mean(lgb_scores)
    
    # 2. XGBoost模型
    print("训练XGBoost模型...")
    xgb_oof, xgb_preds, xgb_scores = cv_model_xgb(X, y, test_x, kf)
    models_oof['xgb'] = xgb_oof
    models_preds['xgb'] = xgb_preds
    models_scores['xgb'] = np.mean(xgb_scores)
    
    # 3. CatBoost模型
    print("训练CatBoost模型...")
    cat_oof, cat_preds, cat_scores = cv_model_cat(X, y, test_x, kf)
    models_oof['cat'] = cat_oof
    models_preds['cat'] = cat_preds
    models_scores['cat'] = np.mean(cat_scores)
    
    # 4. 模型融合
    print("执行模型融合...")
    # 计算每个模型的权重（基于验证集得分）
    total_score = sum(models_scores.values())
    weights = {model: score/total_score for model, score in models_scores.items()}
    
    print(f"模型权重: {weights}")
    
    # 加权融合预测结果
    ensemble_preds = np.zeros(test_x.shape[0])
    for model, preds in models_preds.items():
        ensemble_preds += weights[model] * preds
    
    # 计算融合模型在验证集上的得分
    ensemble_oof = np.zeros(X.shape[0])
    for model, oof in models_oof.items():
        ensemble_oof += weights[model] * oof
    
    ensemble_score = 1/(1+np.sqrt(mean_squared_error(y, ensemble_oof)))
    print(f"融合模型得分: {ensemble_score}")
    
    # 返回融合后的预测结果
    return ensemble_preds

# LightGBM模型训练函数
def cv_model_lgb(train_x, train_y, test_x, kf, seed=2024):
    # 存储验证结果
    oof = np.zeros(train_x.shape[0])
    # 存储测试集预测结果
    test_predict = np.zeros(test_x.shape[0])
    # 存储每折评分
    cv_scores = []
    
    # 优化点7：使用网格搜索寻找最佳参数
    # 定义参数网格
    param_grid = {
        'num_leaves': [31, 63, 127],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_child_weight': [3, 5, 7],
        'feature_fraction': [0.7, 0.8, 0.9],
        'bagging_fraction': [0.7, 0.8, 0.9],
        'lambda_l2': [5, 10, 15]
    }
    
    # 使用第一折数据进行参数搜索
    first_fold = next(kf.split(train_x, train_y))
    train_idx, valid_idx = first_fold
    trn_x, val_x = train_x.iloc[train_idx], train_x.iloc[valid_idx]
    trn_y, val_y = train_y[train_idx], train_y[valid_idx]
    
    # 创建LightGBM数据集
    train_data = lgb.Dataset(trn_x, label=trn_y)
    valid_data = lgb.Dataset(val_x, label=val_y)
    
    # 基础参数
    base_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'bagging_freq': 4,
        'seed': 2023,
        'nthread': 16,
        'verbose': -1,
    }
    
    # 使用网格搜索找到最佳参数
    best_params = base_params.copy()
    best_score = 0
    
    print("执行LightGBM参数搜索...")
    for num_leaves in param_grid['num_leaves']:
        for learning_rate in param_grid['learning_rate']:
            for min_child_weight in param_grid['min_child_weight']:
                for feature_fraction in param_grid['feature_fraction']:
                    for bagging_fraction in param_grid['bagging_fraction']:
                        for lambda_l2 in param_grid['lambda_l2']:
                            params = base_params.copy()
                            params.update({
                                'num_leaves': num_leaves,
                                'learning_rate': learning_rate,
                                'min_child_weight': min_child_weight,
                                'feature_fraction': feature_fraction,
                                'bagging_fraction': bagging_fraction,
                                'lambda_l2': lambda_l2
                            })
                            
                            # 训练模型
                            model = lgb.train(params, train_data, 100, valid_sets=[valid_data], early_stopping_rounds=20, verbose_eval=False)
                            
                            # 计算得分
                            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
                            score = 1/(1+np.sqrt(mean_squared_error(val_y, val_pred)))
                            
                            if score > best_score:
                                best_score = score
                                best_params = params.copy()
    
    print(f"LightGBM最佳参数: {best_params}")
    print(f"LightGBM最佳得分: {best_score}")
    
    # 使用最佳参数进行交叉验证
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print(f'LightGBM折 {i+1}/{kf.n_splits}')
        # 获取当前折的训练集及验证集
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        
        # 使用Lightgbm的Dataset构建训练及验证数据集
        train_matrix = lgb.Dataset(trn_x, label=trn_y)
        valid_matrix = lgb.Dataset(val_x, label=val_y)
        
        # 使用最佳参数训练模型
        model = lgb.train(best_params, train_matrix, 3000, valid_sets=[train_matrix, valid_matrix], early_stopping_rounds=100, verbose_eval=False)
        
        # 对验证集进行预测
        val_pred = model.predict(val_x, num_iteration=model.best_iteration)
        # 对测试集进行预测
        test_pred = model.predict(test_x, num_iteration=model.best_iteration)
        
        # 更新模型在验证集上的结果
        oof[valid_index] = val_pred
        # 将每个模型结果加权并相加
        test_predict += test_pred / kf.n_splits
        
        # 计算得分
        score = 1/(1+np.sqrt(mean_squared_error(val_y, val_pred)))
        # 存储成绩
        cv_scores.append(score)
        print(f"LightGBM折 {i+1} 得分: {score}")
    
    print(f"LightGBM平均得分: {np.mean(cv_scores)}")
    return oof, test_predict, cv_scores

# XGBoost模型训练函数
def cv_model_xgb(train_x, train_y, test_x, kf, seed=2024):
    # 存储验证结果
    oof = np.zeros(train_x.shape[0])
    # 存储测试集预测结果
    test_predict = np.zeros(test_x.shape[0])
    # 存储每折评分
    cv_scores = []
    
    # XGBoost参数
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': seed,
        'nthread': 16
    }
    
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print(f'XGBoost折 {i+1}/{kf.n_splits}')
        # 获取当前折的训练集及验证集
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        
        # 创建DMatrix对象
        dtrain = xgb.DMatrix(trn_x, label=trn_y)
        dvalid = xgb.DMatrix(val_x, label=val_y)
        dtest = xgb.DMatrix(test_x)
        
        # 训练模型
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        model = xgb.train(params, dtrain, 3000, watchlist, early_stopping_rounds=100, verbose_eval=False)
        
        # 对验证集进行预测
        val_pred = model.predict(dvalid)
        # 对测试集进行预测
        test_pred = model.predict(dtest)
        
        # 更新模型在验证集上的结果
        oof[valid_index] = val_pred
        # 将每个模型结果加权并相加
        test_predict += test_pred / kf.n_splits
        
        # 计算得分
        score = 1/(1+np.sqrt(mean_squared_error(val_y, val_pred)))
        # 存储成绩
        cv_scores.append(score)
        print(f"XGBoost折 {i+1} 得分: {score}")
    
    print(f"XGBoost平均得分: {np.mean(cv_scores)}")
    return oof, test_predict, cv_scores

# CatBoost模型训练函数
def cv_model_cat(train_x, train_y, test_x, kf, seed=2024):
    # 存储验证结果
    oof = np.zeros(train_x.shape[0])
    # 存储测试集预测结果
    test_predict = np.zeros(test_x.shape[0])
    # 存储每折评分
    cv_scores = []
    
    # CatBoost参数
    params = {
        'loss_function': 'RMSE',
        'iterations': 3000,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': seed,
        'thread_count': 16,
        'verbose': False
    }
    
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print(f'CatBoost折 {i+1}/{kf.n_splits}')
        # 获取当前折的训练集及验证集
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        
        # 创建CatBoost数据集
        train_pool = cb.Pool(trn_x, label=trn_y)
        valid_pool = cb.Pool(val_x, label=val_y)
        
        # 训练模型
        model = cb.CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=100, verbose=False)
        
        # 对验证集进行预测
        val_pred = model.predict(val_x)
        # 对测试集进行预测
        test_pred = model.predict(test_x)
        
        # 更新模型在验证集上的结果
        oof[valid_index] = val_pred
        # 将每个模型结果加权并相加
        test_predict += test_pred / kf.n_splits
        
        # 计算得分
        score = 1/(1+np.sqrt(mean_squared_error(val_y, val_pred)))
        # 存储成绩
        cv_scores.append(score)
        print(f"CatBoost折 {i+1} 得分: {score}")
    
    print(f"CatBoost平均得分: {np.mean(cv_scores)}")
    return oof, test_predict, cv_scores

# 7.结果输出
def output_results(predictions):
    print("生成预测结果...")
    # 将数据重复4次（因为气象数据是小时级，而功率数据是15分钟级）
    final_predictions = [item for item in predictions for _ in range(4)]
    
    # 读取输出模板
    output = pd.read_csv("/sdc/model/data/output/output1.csv").reset_index(drop=True)
    
    # 添加预测结果
    output["power"] = final_predictions
    
    # 重命名时间列
    output.rename(columns={'Unnamed: 0': ''}, inplace=True)
    
    # 将索引设置为时间列
    output.set_index(output.iloc[:, 0], inplace=True)
    
    # 删掉数据中名为 0 的列
    output = output.drop(columns=["0", ""])
    
    # 存储数据
    output.to_csv('output/output1.csv')
    print("预测结果已保存到 output/output1.csv")
    return output

# 8.主函数
def main():
    # 1. 加载数据
    print("开始加载数据...")
    date_range = pd.date_range(start='2024-01-01', end='2024-12-30')
    date = [date.strftime('%Y%m%d') for date in date_range]
    
    # 加载训练数据
    train_path_template = "/sdc/model/data/初赛训练集/nwp_data_train/1/NWP_1/{}.nc"
    data = [get_data(train_path_template, i) for i in tqdm(date, desc="加载训练数据")]
    train = pd.concat(data, axis=0).reset_index(drop=True)
    
    # 加载目标值
    target = pd.read_csv("/sdc/model/data/初赛训练集/fact_data/1_normalization_train.csv")
    target = target[96:]
    target = target[target['时间'].str.endswith('00:00')]
    target = target.reset_index(drop=True)
    train["power"] = target["功率(MW)"]
    
    # 加载测试数据
    test_path_template = "/sdc/model/data/初赛测试集/nwp_data_test/1/NWP_1/{}.nc"
    test_date = ["20240101", "20240102", "20240103", "20240104", "20240105"]  # 示例测试日期
    test_data = [get_data(test_path_template, i) for i in tqdm(test_date, desc="加载测试数据")]
    test = pd.concat(test_data, axis=0).reset_index(drop=True)
    
    # 2. 数据可视化与分析
    visualize_data(train)
    
    # 3. 数据清洗
    train_cleaned = clean_data(train)
    
    # 4. 特征工程
    train_featured = feature_engineering(train_cleaned)
    test_featured = feature_engineering(test)
    
    # 5. 模型训练与预测
    predictions = train_models(train_featured, test_featured)
    
    # 6. 输出结果
    output = output_results(predictions)
    
    print("处理完成!")
    return output

# 执行主函数
if __name__ == "__main__":
    try:
        import seaborn as sns  # 导入用于绘制热力图的库
        main()
    except Exception as e:
        print(f"执行过程中出现错误: {e}")