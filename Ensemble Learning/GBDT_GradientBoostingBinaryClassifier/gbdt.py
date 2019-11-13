import abc
import math
import logging
import pandas as pd
from decision_tree import Tree
from loss_function import BinomialDeviance
from tree_plot import plot_tree, plot_all_trees

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class AbstractBaseGradientBoosting(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def fit(self, data):
        pass

    def predict(self, data):
        pass


class BaseGradientBoosting(AbstractBaseGradientBoosting):

    def __init__(self, loss, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False):
        super().__init__()
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features = None
        self.trees = {}
        self.f_0 = {}
        self.is_log = is_log
        self.is_plot = is_plot

    def fit(self, data):
        """
        :param data: pandas.DataFrame, the features data of train training   
        """
        # 掐头去尾， 删除id和label，得到特征名称
        self.features = list(data.columns)[1: -1]
        # 初始化 f_0(x)
        # 对于平方损失来说，初始化 f_0(x) 就是 y 的均值
        self.f_0 = self.loss.initialize_f_0(data)
        # 对 m = 1, 2, ..., M
        logger.handlers[0].setLevel(logging.INFO if self.is_log else logging.CRITICAL)
        for iter in range(1, self.n_trees + 1):
            if len(logger.handlers) > 1:
                logger.removeHandler(logger.handlers[-1])
            fh = logging.FileHandler('results/NO.{}_tree.log'.format(iter), mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
            # 计算负梯度--对于平方误差来说就是残差
            logger.info(('-----------------------------构建第%d颗树-----------------------------' % iter))
            self.loss.calculate_residual(data, iter)
            target_name = 'res_' + str(iter)
            self.trees[iter] = Tree(data, self.max_depth, self.min_samples_split,
                                    self.features, self.loss, target_name, logger)
            self.loss.update_f_m(data, self.trees, iter, self.learning_rate, logger)
            if self.is_plot:
                plot_tree(self.trees[iter], max_depth=self.max_depth, iter=iter)
        # print(self.trees)
        if self.is_plot:
            plot_all_trees(self.n_trees)


class GradientBoostingBinaryClassifier(BaseGradientBoosting):
    def __init__(self, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False):
        super().__init__(BinomialDeviance(), learning_rate, n_trees, max_depth,
                         min_samples_split, is_log, is_plot)

    def predict(self, data):
        data['f_0'] = self.f_0
        for iter in range(1, self.n_trees + 1):
            f_prev_name = 'f_' + str(iter - 1)
            f_m_name = 'f_' + str(iter)
            data[f_m_name] = data[f_prev_name] + \
                             self.learning_rate * \
                             data.apply(lambda x: self.trees[iter].root_node.get_predict_value(x), axis=1)
        data['predict_proba'] = data[f_m_name].apply(lambda x: 1 / (1 + math.exp(-x)))
        data['predict_label'] = data['predict_proba'].apply(lambda x: 1 if x >= 0.5 else 0)
