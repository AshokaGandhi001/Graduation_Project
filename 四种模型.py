import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn import tree  # 决策树
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # 用来划分测试集与训练集
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.metrics import explained_variance_score
import warnings
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

# 数据路径
data="/Users/yaoyuechen/Desktop/数据集/one year.csv"
csv_reader = csv.reader(open(data, 'r'))

x = []  # 存储特征数据
x1 = []  # 存储特征数据（备份）
c = []  # 用于存放地震从小到大的震级

# 生成 5.0 到 7.0 的震级列表
for r in np.arange(5.0, 7.1, 0.1):
    c.append('%.1f' % r)
c = list(map(float, c))

y = []  # 存储标签数据
for i in range(21):
    y.append([])

z = []  # 存储地震震级数据
j = 0
for row in csv_reader:
    j += 1
    if j == 1:  # 跳过表头
        continue
    elif 1 < j <= 1089:  # 处理数据行
        row = np.array(row)
        r = list(map(float, row[1:-2]))  # 提取特征数据
        x.append(r)
        x1.append(r)
        z.append(float(row[-2]))  # 提取地震震级

# 数据预处理
x = scale(x)  # 标准化特征数据

# 打印总样本数
print(f"总样本数: {len(z)}")

# 根据不同的震级阈值生成标签数据
for i in range(21):
    for j in range(len(z)):
        if z[j] < c[i]:
            s = 0  # 小于震级阈值的设置为 0
        else:
            s = 1  # 大于等于震级阈值的设置为 1
        y[i].append(s)

        # 打印每个阈值下的正负样本数量
        print(f"阈值 {c[i]}: 正样本数量 {sum(y[i])}, 负样本数量 {len(y[i]) - sum(y[i])}")

# 检查类别数量是否小于 2，如果是，则添加一个额外的类别
unique_classes = np.unique(y[i])
if len(unique_classes) < 2:
    y[i].append(1 - unique_classes[0])

# 初始化存储结果的列表
acc, pre, rec, f_1 = [], [], [], []
for i in range(4):
    acc.append([])
    pre.append([])
    rec.append([])
    f_1.append([])

# 定义评估函数
def calculate_metrics(y_true, y_pred):
    tp = sum(1 for k in range(len(y_true)) if y_true[k] == y_pred[k] == 1)
    tn = sum(1 for k in range(len(y_true)) if y_true[k] == y_pred[k] == 0)
    fn = sum(1 for k in range(len(y_true)) if y_true[k] == 1 and y_pred[k] == 0)
    fp = sum(1 for k in range(len(y_true)) if y_true[k] == 0 and y_pred[k] == 1)

    accuracy = (tp + tn) / len(y_true)
    recall = tp / (tp + fn) if (tp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return accuracy, recall, precision, f1

# 定义感兴趣的震级阈值
thresholds_of_interest = [5.0, 5.5, 6.0, 6.5, 7.0]
rf_results = []  # 用于存储随机森林的结果

# 模型训练与评估
for i in range(21):
    x_train, x_test, y_train, y_test = train_test_split(x, y[i], test_size=0.2, random_state=66)  # 数据划分
    x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x1, y[i], test_size=0.2, random_state=66)
    np.unique(y_train)
    if __name__ == '__main__':
        # 定义分类器模型
        model = [svm.SVC(kernel='rbf', C=1),  # 支持向量机
                 LogisticRegression(),  # 逻辑回归
                 ]
        model_1 = [tree.DecisionTreeClassifier(criterion='gini'), RandomForestClassifier()]
        print("当震级为 " + str(c[i]) + " 时的预测结果")
        for j in range(2):
            model[j].fit(x_train, y_train)
            model_1[j].fit(x_train_1, y_train_1)
            pred = model[j].predict(x_test)
            pred_1 = model_1[j].predict(x_test_1)
            warnings.filterwarnings("ignore")  # 忽略警告

            # 计算指标
            accuracy, recall, precision, f1_score = calculate_metrics(y_test, pred)
            accuracy_1, recall_1, precision_1, f1_score_1 = calculate_metrics(y_test_1, pred_1)

            # 计算正样本中被预测为正的数量和负样本中被预测为负的数量
            true_positive = sum(1 for k in range(len(y_test_1)) if y_test_1[k] == 1 and pred_1[k] == 1)
            true_negative = sum(1 for k in range(len(y_test_1)) if y_test_1[k] == 0 and pred_1[k] == 0)

            acc[j].append(accuracy)
            pre[j].append(precision)
            rec[j].append(recall)
            f_1[j].append(f1_score)

            acc[j+2].append(accuracy_1)
            pre[j+2].append(precision_1)
            rec[j+2].append(recall_1)
            f_1[j+2].append(f1_score_1)

            # 如果当前模型是随机森林，且震级阈值在感兴趣的范围内，记录结果
            if j == 1 and c[i] in thresholds_of_interest:
                # 计算测试集中的正负样本数量
                positive_samples = sum(y_test_1)
                negative_samples = len(y_test_1) - positive_samples

                # 记录结果
                rf_results.append({
                    'Threshold': c[i],
                    'Accuracy': accuracy_1,
                    'Precision': precision_1,
                    'Recall': recall_1,
                    'F1 Score': f1_score_1,
                    'Positive Samples': positive_samples,
                    'Negative Samples': negative_samples,
                    'True Positives': true_positive,
                    'True Negatives': true_negative
                })


# 将随机森林的结果转换为 DataFrame
import pandas as pd

# 设置显示所有列
pd.set_option('display.max_columns', None)
rf_results_df = pd.DataFrame(rf_results)
print("\n随机森林在指定阈值下的指标：")
print(rf_results_df)


# 绘制指标曲线
plt.figure(1)
for i in range(4):
    plt.plot(c, acc[i], linewidth=2)
plt.xticks(np.arange(5.0, 7.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(['Support Vector Machine', 'Logistic Regression', 'Decision Tree', 'Random Forest'], loc='best')
plt.xlabel("Magnitude", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.title('Accuracy', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.5)

plt.figure(2)
for i in range(4):
    plt.plot(c, pre[i], linewidth=2)
plt.xticks(np.arange(5.0, 7.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(['Support Vector Machine', 'Logistic Regression', 'Decision Tree', 'Random Forest'], loc='best')
plt.xlabel("Magnitude", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.title('Precision', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.5)

plt.figure(3)
for i in range(4):
    plt.plot(c, rec[i], linewidth=2)
plt.xticks(np.arange(5.0, 7.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(['Support Vector Machine', 'Logistic Regression', 'Decision Tree', 'Random Forest'], loc='best')
plt.xlabel("Magnitude", fontsize=14)
plt.ylabel("Recall", fontsize=14)
plt.title('Recall', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.5)

plt.figure(4)
for i in range(4):
    plt.plot(c, f_1[i], linewidth=2)
plt.xticks(np.arange(5.0, 7.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(['Support Vector Machine', 'Logistic Regression', 'Decision Tree', 'Random Forest'], loc='best')
plt.xlabel("Magnitude", fontsize=14)
plt.ylabel("F1", fontsize=14)
plt.title('F1 Score', fontsize=24)
plt.grid(True, linestyle='--', alpha=0.5)

# 绘制子图
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
for i in range(4):
    plt.plot(c, acc[i], linewidth=2)
plt.xticks(np.arange(5.0, 7.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend(['Support Vector Machine', 'Logistic Regression', 'Decision Tree', 'Random Forest'],
           loc='best', prop={'size': 13})
plt.ylabel("Accuracy", fontsize=18)
plt.title('(a)  Accuracy', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(2, 2, 2)
for i in range(4):
    plt.plot(c, pre[i], linewidth=2)
plt.xticks(np.arange(5.0, 7.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylabel("Precision", fontsize=18)
plt.title('(b)  Precision', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(2, 2, 3)
for i in range(4):
    plt.plot(c, rec[i], linewidth=2)
plt.xticks(np.arange(5.0, 7.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel("Magnitude", fontsize=14)
plt.ylabel("Recall", fontsize=18)
plt.title('(c)  Recall', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.5)

plt.subplot(2, 2, 4)
for i in range(4):
    plt.plot(c, f_1[i], linewidth=2)
plt.xticks(np.arange(5.0, 7.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel("Magnitude", fontsize=14)
plt.ylabel("F1", fontsize=18)
plt.title('(d)  F1 Score', fontsize=18)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
