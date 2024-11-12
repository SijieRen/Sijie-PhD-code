import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import copy
from sklearn.model_selection import train_test_split
print(os.getcwd())
# 加载Excel文件
file_path = './data/stas_前期数据.xls'  # 替换为你的Excel文件路径

# 读取训练集和测试集
train_df = pd.read_excel(file_path, sheet_name='inner')
# test1_df = pd.read_excel(file_path, sheet_name='exter')
test2_df = pd.read_excel(file_path, sheet_name='exter')

# 假设x列的名称为'feature', y列的名称为'label'
X_train = train_df[['gender', 'spiculated', 'lobulated', 'cavity', 'vacuole', 'boundary', 'airbronchogram', 'vpi', 'vessel', 'lymphadenovarix', 'max_real']]  # 替换为实际的列名
y_train = train_df['y']  # 替换为实际的列名

# 由于我们有多个测试集，我们将它们合并为一个
# X_test1 = test1_df[['sex (1=F 0=M)', 'age', 'type_solid', 'type_mGGN', 'type_pGGN', 'spiculated', 'lobulated', 'cavity', 'vacuole', 'boundary', 'Air bronchogram', 'Pleural indentation', 'Pulmonary vessel', 'Maxi dia_grade1', 'Maxi dia_grade2', 'Maxi dia_grade3']]  # 替换为实际的列名
# y_test1 = test1_df['y']  # 替换为实际的列名

X_train, X_test1, y_train, y_test1 = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

X_test2 = test2_df[['gender', 'spiculated', 'lobulated', 'cavity', 'vacuole', 'boundary', 'airbronchogram', 'vpi', 'vessel', 'lymphadenovarix', 'max_real']]  # 替换为实际的列名
y_test2 = test2_df['y']  # 替换为实际的列名

# 将标签转换为二进制格式，因为我们需要计算AUC
y_train_binarized = label_binarize(y_train, classes=np.unique(y_train))
n_classes = y_train_binarized.shape[1]



# 定义参数范围
# param_distributions = {
#     'n_estimators': np.arange(10, 101, 10),
#     'max_depth': [None] + list(np.arange(5, 21, 5)),
#     'min_samples_split': np.arange(2, 11),
#     'min_samples_leaf': np.arange(1, 11),
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'bootstrap': [True, False]
# }
# 创建RandomizedSearchCV对象
# random_search = RandomizedSearchCV(
#     estimator=rf,
#     param_distributions=param_distributions,
#     n_iter=100,  # 随机选择参数组合的数量
#     cv=3,  # 交叉验证的折数
#     verbose=2,  # 输出详细的日志信息
#     random_state=42,
#     n_jobs=-1  # 使用所有可用的处理器
# )


# 定义参数网格
param_grid = {
    'n_estimators': [10, 50,80, 100],  # 树的数量
    'max_depth': [None, 10, 20, 30],     # 树的最大深度
    'min_samples_split': [2, 5, 10],     # 分割内部节点所需的最小样本数
    'min_samples_leaf': [1, 2, 4],       # 叶节点所需的最小样本数
    'max_features': ['sqrt', 'log2'],  # 寻找最佳分割时要考虑的特征数量
    'bootstrap': [True, False]           # 是否使用bootstrap样本
}


best_test1_auc = 0
best_test1_acc = 0
best_test2_auc = 0
best_test2_acc = 0
y_pred_test1_best = []
y_pred_test2_best = []
best_test1_param = []
best_test2_param = []


for i in range(4):
    for j in range(4):
        for k in range(3):
            for l in range(3):
                for m in range(2):
                    for n in range(2):
                        print(f"iteration: {(i+1)*(j+1)*(k+1)*(l+1)*(m+1)*(n+1)}:")
                        rf = RandomForestClassifier(n_estimators = param_grid["n_estimators"][i],
                                                    max_depth=param_grid["max_depth"][j],
                                                    min_samples_split=param_grid["min_samples_split"][k],
                                                    min_samples_leaf=param_grid["min_samples_leaf"][l],
                                                    max_features=param_grid["max_features"][m],
                                                    bootstrap=param_grid["bootstrap"][n])
                        # 定义随机森林分类器
                        rf.fit(X_train, y_train)        
                        y_pred_test1 = rf.predict(X_test1)
                        y_pred_test2 = rf.predict(X_test2)
                        auc_scores_test1 = roc_auc_score(y_test1, y_pred_test1)
                        auc_scores_test2 = roc_auc_score(y_test2, y_pred_test2)
                        accuracy_test1 = accuracy_score(y_test1, y_pred_test1)
                        accuracy_test2 = accuracy_score(y_test2, y_pred_test2)

                        if auc_scores_test1 > best_test1_auc:
                            best_test1_auc = copy.deepcopy(auc_scores_test1)
                            best_test1_acc = copy.deepcopy(accuracy_test1)
                            y_pred_test1_best = copy.deepcopy(y_pred_test1)
                            best_test1_param = copy.deepcopy(rf.get_params())

                        if auc_scores_test2 > best_test2_auc:
                            best_test2_auc = copy.deepcopy(auc_scores_test2)
                            best_test2_acc = copy.deepcopy(accuracy_test2)
                            y_pred_test2_best = copy.deepcopy(y_pred_test2)
                            best_test2_param = copy.deepcopy(rf.get_params())


# # 初始化随机森林分类器
# rf = RandomForestClassifier(n_estimators=10, max_leaf_nodes=42,max_depth=10, max_features=10, random_state=42)

# # 训练模型
# rf.fit(X_train, y_train)

# 预测测试集


# 计算AUC和ACC
# auc_scores_test1 = [roc_auc_score(fpr=y_test1[:, i], tpr=y_pred_test1[:, i]) for i in range(n_classes)]
# auc_scores_test2 = [roc_auc_score(fpr=y_test2[:, i], tpr=y_pred_test2[:, i]) for i in range(n_classes)]


# # 找到AUC性能最好的模型参数
# best_auc_score = max(auc_scores_test1 + auc_scores_test2)
# best_auc_index = auc_scores_test1.index(max(auc_scores_test1)) if max(auc_scores_test1) >= max(auc_scores_test2) else len(auc_scores_test1)




# 输出模型参数
# print(f"Model Parameters: {rf.get_params()}")

from sklearn.metrics import confusion_matrix, classification_report

# 为了计算sensitivity和specificity，我们首先需要二进制标签
# 由于我们之前使用了label_binarize，我们假设现在是二分类问题

# 计算Test Set 1的混淆矩阵
cm_test1 = confusion_matrix(y_test1, y_pred_test1_best)

# 计算Test Set 2的混淆矩阵
cm_test2 = confusion_matrix(y_test2, y_pred_test2_best)

# 计算Test Set 1的sensitivity和specificity
tn, fp, fn, tp = cm_test1.ravel()
sensitivity_test1 = tp / (tp + fn)
specificity_test1 = tn / (tn + fp)

# 计算Test Set 2的sensitivity和specificity
tn, fp, fn, tp = cm_test2.ravel()
sensitivity_test2 = tp / (tp + fn)
specificity_test2 = tn / (tn + fp)

print("Inner***"*10)
print(f"Best AUC Score_inter: {best_test1_auc}")
print(f"Test Set 1 Sensitivity: {sensitivity_test1}")
print(f"Test Set 1 Specificity: {specificity_test1}")
print(f"Corresponding ACC on Test_inter: {best_test1_acc}")
print(f"Best parameter for Test_inter: {best_test1_param}")
print("Inner***"*10)
print("------------------------------------------------")
print("Exter***"*10)
print(f"Best AUC Score_exter: {best_test2_auc}")
print(f"Test Set 2 Sensitivity: {sensitivity_test2}")
print(f"Test Set 2 Specificity: {specificity_test2}")
print(f"Corresponding ACC on Test_exter: {best_test2_acc}")
print(f"Best parameter for Test_inter: {best_test2_param}")
print("Exter***"*10)


# 你也可以使用classification_report来获取更详细的性能指标
print(classification_report(y_test1, y_pred_test1_best))
print(classification_report(y_test2, y_pred_test2_best))