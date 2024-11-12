import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import os
print(os.getcwd())
# 加载Excel文件
file_path = './Causality/zhongshan_lung/DarMo_supplementary/data/STAS_dataset_2D.xls'  # 替换为你的Excel文件路径

# 读取训练集和测试集
train_df = pd.read_excel(file_path, sheet_name='train')
test1_df = pd.read_excel(file_path, sheet_name='test_inter')
test2_df = pd.read_excel(file_path, sheet_name='test_exter')

# 假设x列的名称为'feature', y列的名称为'label'
X_train = train_df[['sex (1=F 0=M)', 'age', 'type_solid', 'type_mGGN', 'type_pGGN', 'spiculated', 'lobulated', 'cavity', 'vacuole', 'boundary', 'Air bronchogram', 'Pleural indentation', 'Pulmonary vessel', 'Maxi dia_grade1', 'Maxi dia_grade2', 'Maxi dia_grade3']]  # 替换为实际的列名
y_train = train_df['y']  # 替换为实际的列名

# 由于我们有多个测试集，我们将它们合并为一个
X_test1 = test1_df[['sex (1=F 0=M)', 'age', 'type_solid', 'type_mGGN', 'type_pGGN', 'spiculated', 'lobulated', 'cavity', 'vacuole', 'boundary', 'Air bronchogram', 'Pleural indentation', 'Pulmonary vessel', 'Maxi dia_grade1', 'Maxi dia_grade2', 'Maxi dia_grade3']]  # 替换为实际的列名
y_test1 = test1_df['y']  # 替换为实际的列名
X_test2 = test2_df[['sex (1=F 0=M)', 'age', 'type_solid', 'type_mGGN', 'type_pGGN', 'spiculated', 'lobulated', 'cavity', 'vacuole', 'boundary', 'Air bronchogram', 'Pleural indentation', 'Pulmonary vessel', 'Maxi dia_grade1', 'Maxi dia_grade2', 'Maxi dia_grade3']]  # 替换为实际的列名
y_test2 = test2_df['y']  # 替换为实际的列名

# 将标签转换为二进制格式，因为我们需要计算AUC
y_train_binarized = label_binarize(y_train, classes=np.unique(y_train))
n_classes = y_train_binarized.shape[1]

# 初始化随机森林分类器
rf = RandomForestClassifier(n_estimators=10, max_leaf_nodes=42,max_depth=10, max_features=10, random_state=42)

# 训练模型
rf.fit(X_train, y_train)

# 预测测试集
y_pred_test1 = rf.predict(X_test1)
y_pred_test2 = rf.predict(X_test2)

# 计算AUC和ACC
# auc_scores_test1 = [roc_auc_score(fpr=y_test1[:, i], tpr=y_pred_test1[:, i]) for i in range(n_classes)]
# auc_scores_test2 = [roc_auc_score(fpr=y_test2[:, i], tpr=y_pred_test2[:, i]) for i in range(n_classes)]
auc_scores_test1 = [roc_auc_score(y_test1, y_pred_test1)]
auc_scores_test2 = [roc_auc_score(y_test2, y_pred_test2)]
accuracy_test1 = accuracy_score(y_test1, y_pred_test1)
accuracy_test2 = accuracy_score(y_test2, y_pred_test2)

# 找到AUC性能最好的模型参数
best_auc_score = max(auc_scores_test1 + auc_scores_test2)
best_auc_index = auc_scores_test1.index(max(auc_scores_test1)) if max(auc_scores_test1) >= max(auc_scores_test2) else len(auc_scores_test1)

print(f"Best AUC Score_inter: {auc_scores_test1}")
print(f"Best AUC Score_exter: {auc_scores_test2}")
print(f"Corresponding ACC on Test_inter: {accuracy_test1}")
print(f"Corresponding ACC on Test_exter: {accuracy_test2}")

# 输出模型参数
print(f"Model Parameters: {rf.get_params()}")

from sklearn.metrics import confusion_matrix, classification_report

# 为了计算sensitivity和specificity，我们首先需要二进制标签
# 由于我们之前使用了label_binarize，我们假设现在是二分类问题

# 计算Test Set 1的混淆矩阵
cm_test1 = confusion_matrix(y_test1, y_pred_test1)

# 计算Test Set 2的混淆矩阵
cm_test2 = confusion_matrix(y_test2, y_pred_test2)

# 计算Test Set 1的sensitivity和specificity
tn, fp, fn, tp = cm_test1.ravel()
sensitivity_test1 = tp / (tp + fn)
specificity_test1 = tn / (tn + fp)

# 计算Test Set 2的sensitivity和specificity
tn, fp, fn, tp = cm_test2.ravel()
sensitivity_test2 = tp / (tp + fn)
specificity_test2 = tn / (tn + fp)

print(f"Test Set 1 Sensitivity: {sensitivity_test1}")
print(f"Test Set 1 Specificity: {specificity_test1}")

print(f"Test Set 2 Sensitivity: {sensitivity_test2}")
print(f"Test Set 2 Specificity: {specificity_test2}")

# 你也可以使用classification_report来获取更详细的性能指标
print(classification_report(y_test1, y_pred_test1))
print(classification_report(y_test2, y_pred_test2))