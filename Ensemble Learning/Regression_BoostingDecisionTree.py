# coding=utf-8
import numpy as np

label = np.array([5.56, 5.7, 5.91, 6.4, 6.8, 7.05, 8.9, 8.7, 9, 9.05])

# 已经排好序了。实际情况中单一特征的数据或者多特征的数据，选择切分点的时候也像决策树一样选择
feature = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class Tree_model:
    def __init__(self, stump, mse, left_value, right_value, residual):
        '''
        :param stump: 为feature最佳切割点
        :param mse: 为每棵树的平方误差
        :param left_value: 为决策树左值
        :param right_value: 为决策树右值
        :param residual: 为每棵决策树生成后余下的残差
        '''
        self.stump = stump
        self.mse = mse
        self.left_value = left_value
        self.right_value = right_value
        self.residual = residual


'''根据feature准备好切分点。例如:
feature为[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
切分点为[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
'''


def Get_stump_list(feature):
    # 特征值从小到大排序好,错位相加
    tmp1 = list(feature.copy())
    tmp2 = list(feature.copy())
    tmp1.insert(0, 0)
    tmp2.append(0)
    stump_list = ((np.array(tmp1) + np.array(tmp2)) / float(2))[1:-1]
    return stump_list


# 此处的label其实是残差
def Get_decision_tree(stump_list, feature, label):
    best_mse = np.inf
    best_stump = 0  # min(stump_list)
    residual = np.array([])
    left_value = 0
    right_value = 0
    for i in range(np.shape(stump_list)[0]):
        left_node = []
        right_node = []
        for j in range(np.shape(feature)[0]):
            if feature[j] < stump_list[i]:
                left_node.append(label[j])
            else:
                right_node.append(label[j])
        left_mse = np.sum((np.average(left_node) - np.array(left_node)) ** 2)
        right_mse = np.sum((np.average(right_node) - np.array(right_node)) ** 2)
        # print("decision stump: %d, left_mse: %f, right_mse: %f, mse: %f" % (i, left_mse, right_mse, (left_mse + right_mse)))
        if best_mse > (left_mse + right_mse):
            best_mse = left_mse + right_mse
            left_value = np.average(left_node)
            right_value = np.average(right_node)
            best_stump = stump_list[i]
            left_residual = np.array(left_node) - left_value
            right_residual = np.array(right_node) - right_value
            residual = np.append(left_residual, right_residual)
            # print("decision stump: %d, residual: %s"% (i, residual))
    Tree = Tree_model(best_stump, best_mse, left_value, right_value, residual)
    return Tree, residual


# Tree_num就是树的数量
def BDT_model(feature, label, Tree_num=100):
    feature = np.array(feature)
    label = np.array(label)
    stump_list = Get_stump_list(feature)
    Trees = []
    residual = label.copy()
    # 产生每一棵树
    for num in range(Tree_num):
        # 每次新生成树后，还需要再次更新残差residual
        Tree, residual = Get_decision_tree(stump_list, feature, residual)
        Trees.append(Tree)
    return Trees


def BDT_predict(Trees, feature):
    predict_list = [0 for i in range(np.shape(feature)[0])]
    # 将每棵树对各个特征预测出来的结果进行相加，相加的最后结果就是最后的预测值
    for Tree in Trees:
        for i in range(np.shape(feature)[0]):
            if feature[i] < Tree.stump:
                predict_list[i] = predict_list[i] + Tree.left_value
            else:
                predict_list[i] = predict_list[i] + Tree.right_value
    return predict_list


# 计算误差
def Get_error(predict, label):
    predict = np.array(predict)
    label = np.array(label)
    error = np.sum((label - predict) ** 2)
    return error


Trees = BDT_model(feature, label)
predict = BDT_predict(Trees, feature)
print("The error is ", Get_error(predict, label))
print(predict)