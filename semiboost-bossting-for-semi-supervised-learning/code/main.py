from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from semiboost_svm import SemiBoostSvm
import math

def sign(p, q):
    if p > q:
        return 1
    else:
        return -1

def binary_encode(str):
    if str.decode('utf-8') == 'M':
        return -1
    else:
        return 1

print('读取数据 ... ')
# read the dataset, with the format (label, feature_0, feature_1, feature_2 , ...  , feature_29)
dataset = np.loadtxt(fname='./dataset/WDBC.dat', usecols=range(1,32), delimiter=',', converters={1: binary_encode})

# apply PCA to the dataset
# todo
print('分割训练集和测试集 ... ')
# split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(dataset[:, 1:], dataset[:, 0], test_size=0.5)

# label 10 of the training set and delete the label of the rest of the training set
y_train[10:] = np.zeros(y_train.size - 10)
training_set = np.c_[X_train, y_train]

# the number of iterations
T = 20
print('迭代次数设置为%d' % (T))

# the number of labeled samples
n_labeled = 10

# the number of unlabeled samples
n_unlabeled = y_train.size - 10

sb_svm = SemiBoostSvm()

for t in range(T):
    print('第%d轮迭代 ... ' % (t))
    # calculate the confidence
    unlabeled_training_set = training_set[training_set[:,-1] == 0, :]
    labeled_training_set = training_set[training_set[:,-1] != 0, :]
    S_uu = cosine_similarity(unlabeled_training_set) # 未标记示例的相似矩阵
    S_ul = cosine_similarity(unlabeled_training_set, labeled_training_set) # 未标记示例和已标记示例的相似矩阵

    # set C as the ratio: n_l / n_c
    C = n_labeled / n_unlabeled

    positive_confidences = []
    negative_confidences = []

    # 计算未标记数据的置信度
    for i in range(n_unlabeled):
        positive_former_term = 0
        positive_latter_term = 0
        negative_former_term = 0
        negative_latter_term = 0
        for j in range(n_labeled):
            if labeled_training_set[j][-1] == 1:
                positive_former_term += S_ul[i][j] * math.exp(-2 * sb_svm.predict(unlabeled_training_set[i, 0:-1]))
            else:
                negative_former_term += S_ul[i][j] * math.exp(2 * sb_svm.predict(unlabeled_training_set[i, 0:-1]))
        for j in range(n_unlabeled):
            positive_latter_term += S_uu[i][j] * math.exp(sb_svm.predict(unlabeled_training_set[j, 0:-1]) - sb_svm.predict(unlabeled_training_set[i, 0:-1]))
            negative_latter_term += S_uu[i][j] * math.exp(sb_svm.predict(unlabeled_training_set[i, 0:-1]) - sb_svm.predict(unlabeled_training_set[j, 0:-1]))
        positive_confidences.append(positive_former_term + C / 2 * positive_latter_term)
        negative_confidences.append(negative_former_term + C / 2 * negative_latter_term)

    # select the top 10% of the unlabeled samples
    absolute_confidences = np.absolute(np.array(positive_confidences) - np.array(negative_confidences))
    confidence_dict = dict(zip(range(n_unlabeled), absolute_confidences))
    selected_unlabeled_tuples = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
    n_selected = math.ceil(n_unlabeled * 0.1)
    
    selected_unlabeled_tuples = selected_unlabeled_tuples[slice(n_selected)]

    for i in range(n_selected): # 分配伪标签
        index_unlabeled = selected_unlabeled_tuples[i][0]
        if positive_confidences[index_unlabeled] >= negative_confidences[index_unlabeled]:
            unlabeled_training_set[index_unlabeled][-1] = 1
        else:
            unlabeled_training_set[index_unlabeled][-1] = -1

    training_set = np.r_[labeled_training_set, unlabeled_training_set]
    labeled_training_set = training_set[training_set[:,-1] != 0, :]

    # train SVM
    component_classifier = svm.LinearSVC() # 引入sklearn内置的SVM分类器
    component_classifier.fit(labeled_training_set[:, 0:-1], labeled_training_set[:, -1:])

    # calculate \alpha_t to check if it's negative
    # \alpha_t即接下来的weight，为组合分类器时的分类器权重
    weighted_error_denominator = 0
    weighted_error_numerator = 0
    for i in range(n_unlabeled):
        # todo
        weighted_error_denominator += positive_confidences[i] + negative_confidences[i]
        if sign(positive_confidences[i], negative_confidences[i]) == -1:
            weighted_error_numerator += positive_confidences[i]
        else:
            weighted_error_numerator += negative_confidences[i]

    weighted_error = weighted_error_numerator / weighted_error_denominator
    weight = 1 / 4 * np.log((1 - weighted_error) / weighted_error)

    if weight < 0: # 达成结束循环的条件
        break

    sb_svm.boost(component_classifier, weight)
    n_unlabeled -= n_selected
    pass

error_count = 0
for i in range(X_test.shape[0]):
    if sb_svm.predict(X_test[i]) != y_test[i]:
        error_count += 1

print('The accuracy is: ')
print(1 - error_count / X_test.shape[0])