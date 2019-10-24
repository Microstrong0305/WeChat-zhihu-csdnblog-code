import pandas as pd
import shutil
from GBDT.gbdt import GradientBoostingRegressor


def run(args):
    # 训练数据和测试数据
    train_data = pd.DataFrame(data=[[1, 5, 20, 1.1],
                                    [2, 7, 30, 1.3],
                                    [3, 21, 70, 1.7],
                                    [4, 30, 60, 1.8],
                                    ], columns=['id', 'age', 'weight', 'label'])

    test_data = pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])
    # 创建模型结果的目录
    if not os.path.exists('results'):
        os.makedirs('results')
    if len(os.listdir('results')) > 0:
        shutil.rmtree('results')
        os.makedirs('results')
    # 初始化模型
    model = GradientBoostingRegressor(learning_rate=args.lr, n_trees=args.trees, max_depth=args.depth,
                                      min_samples_split=args.count, is_log=args.log, is_plot=args.plot)
    # 训练模型
    model.fit(train_data)
    # 记录日志
    logger.removeHandler(logger.handlers[-1])
    logger.addHandler(logging.FileHandler('results/result.log'.format(iter), mode='w', encoding='utf-8'))
    logger.info(data)
    # 模型预测
    model.predict(test_data)
    # 记录日志
    logger.setLevel(logging.INFO)
    logger.info((test_data['predict_value']))
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GBDT-Regression-Tutorial')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--trees', default=5, type=int, help='the number of decision trees')
    parser.add_argument('--depth', default=3, type=int, help='the max depth of decision trees')
    # 非叶节点的最小数据数目，如果一个节点只有一个数据，那么该节点就是一个叶子节点，停止往下划分
    parser.add_argument('--count', default=2, type=int, help='the min data count of a node')
    parser.add_argument('--log', default=False, type=bool, help='whether to print the log on the console')
    parser.add_argument('--plot', default=True, type=bool, help='whether to plot the decision trees')
    args = parser.parse_args()
    run(args)
