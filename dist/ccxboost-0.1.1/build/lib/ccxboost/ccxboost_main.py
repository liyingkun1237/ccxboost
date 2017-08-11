'''
调用语句

设计思路，第一步先写为函数，后面抽象为类

#1.加载数据
    |--train.csv
    |--test.csv
    |--指明主键，指明目标变量
#2.网格搜索+交叉验证
    |解析配置文件内容
    |开始运行
    |中间结果保存
    |最优参数产生
#3.最优模型训练+重要变量
#4.模型预测+模型评价+模型结果可视化
'''
from ccxboost.model_function import load_data, model_data, model_cv, save_data, get_bstpram, model_train, \
    get_importance_var, save_bstmodel, model_predict, plot_ks_line, plot_roc_line, get_modelpredict_re, plot_imp, \
    write_path
import configparser
from ccxboost.paserconf import extract_conf
import os


def ccxboost_main(train_path, test_path, index_name, target_name):
    # 1.读取数据
    train = load_data(train_path)  # .select_dtypes(exclude=['object'])
    test = load_data(test_path)  # .select_dtypes(exclude=['object'])

    object_var = list(train.select_dtypes(include=['object']).columns.values)
    warn_col = [x for x in object_var if x not in [index_name]]
    if warn_col:
        print('数据中列名为%s的列,不是数值型数据,请转换为数值型数据或删除后再输入.' % warn_col)

    del_col = [index_name, target_name] + warn_col
    x_colnames = [x for x in train.columns if x not in del_col]
    y_colnames = target_name

    # 2.转换数据格式为模型要求格式
    dtrain = model_data(train, x_colnames, y_colnames)
    dtest = model_data(test, x_colnames, y_colnames)

    # 解析配置文件，获取网格搜索的调参列表
    conf_path = os.path.split(os.path.realpath(__file__))[0] + '/ccxboost.conf'
    # print('###########', conf_path)
    param_grid = extract_conf(conf_path, 'XGB_PARAMS')

    # 用config对象读取配置文件，获取到交叉验证的option参数
    conf = configparser.ConfigParser()
    conf.read(conf_path)
    num_boost_rounds = conf.getint("XGB_OPTIONS", "num_round")
    # nthread = conf.getint("XGB_OPTIONS", "nthread")
    cv = conf.getint("XGB_OPTIONS", "cv")
    cv_mess = conf.get("XGB_OPTIONS", "cv_mess")

    # 网格搜索
    re = model_cv(dtrain, param_grid, num_boost_rounds, nfold=cv, message=cv_mess)
    file_name = cv_mess + '_' + str(cv) + 'FlodCV.csv'
    cv_result_path = save_data(re, file_name, index=True)

    param, num_round = get_bstpram(re)

    bst = model_train(dtrain, dtest, param, num_round)
    model_path = save_bstmodel(bst, cv_mess)
    # bst.dump_model('bst_model.txt')

    # 重要变量
    imp_var = get_importance_var(bst)
    # plot_imp(bst)
    imp_path = save_data(imp_var, 'importance_var.csv')

    # 模型预测与模型评估
    train_pred_y, test_pred_y = model_predict(bst, dtrain, dtest, message=cv_mess)
    # 模型预测结果
    pred_path = save_data(get_modelpredict_re(test[index_name], test_pred_y), 'test_predict.csv')
    # 画图
    trks_path = plot_ks_line(dtrain.get_label(), train_pred_y, title='train_ks-line')
    trauc_path = plot_roc_line(dtrain.get_label(), train_pred_y, title='train_ROC-line')
    # 注意，现在仅支持测试集有目标变量的，没有的情况需要后期优化时注意
    teks_path = plot_ks_line(dtest.get_label(), test_pred_y, title='test_ks-line')
    teauc_path = plot_roc_line(dtest.get_label(), test_pred_y, title='test_ROC-line')

    path_list = [cv_result_path, model_path, imp_path, pred_path, trks_path, trauc_path, teks_path, teauc_path]
    file = r'/opt/django_ML/static/modelpath.txt'
    write_path(file, path_list)
    # print(path_list)

if __name__ == '__main__':
    train_path = r'/opt/django_ML/upload/train_base14.csv'
    test_path = r'/opt/django_ML/upload/test_base14.csv'
    index_name = 'contract_id'
    target_name = 'target'

    ccxboost_main(train_path, test_path, index_name, target_name)







# 这部分应该柴实现，读取用户长传的数据，并获取到用户指定的索引，用户指定的目标变量，我得到的就是一个可用的数据集
# 暂时使用我简化版的load_data替代


# 数据加载
# train_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\train_base14.csv'
# test_path = r'C:\Users\liyin\Desktop\20170620_tn\0620_base\test_base14.csv'
# train = load_data(train_path)
# test = load_data(test_path)

# 字段信息获取与确定
# del_col = ['tg_location1', 'contract_id', 'target']
# x_colnames = [x for x in train.columns if x not in del_col]
# y_colnames = ['target']

# 转换数据格式为模型要求格式
# dtrain = model_data(train, x_colnames, y_colnames)
# dtest = model_data(test, x_colnames, y_colnames)

# 解析配置文件，获取网格搜索的调参列表
# param_grid = extract_conf(r"C:\Users\liyin\Desktop\ccxboost\ccxboost\ccxboost.conf", 'XGB_PARAMS')

# 用config对象读取配置文件，获取到交叉验证的option参数
# conf = configparser.ConfigParser()
# conf.read(r"C:\Users\liyin\Desktop\ccxboost\ccxboost\ccxboost.conf")
# num_boost_rounds = conf.getint("XGB_OPTIONS", "num_round")
# nthread = conf.getint("XGB_OPTIONS", "nthread")
# cv = conf.getint("XGB_OPTIONS", "cv")
# cv_mess = conf.get("XGB_OPTIONS", "cv_mess")

# 网格搜索
# re = model_cv(dtrain, param_grid, num_boost_rounds, nfold=cv, message=cv_mess)
# save_data(re, 'model_cv_1_basemobile.csv', index=True)
#
# param, num_round = get_bstpram(re)

# bst = model_train(dtrain, dtest, param, num_round)
# save_bstmodel(bst)
# bst.dump_model('bst_model.txt')
# 重要变量
# imp_var = get_importance_var(bst)
# plot_imp(bst)
# save_data(imp_var, 'imp_var_basemobile.csv', index=True)

# 模型预测与模型评估
# train_pred_y, test_pred_y = model_predict(bst, dtrain, dtest, message='data_id_basemobile_1')
# 模型预测结果
# get_modelpredict_re(test.contract_id, test_pred_y)
# 画图
# plot_ks_line(dtest.get_label(), test_pred_y, title='black_basemobile_ks-line')
# plot_roc_line(dtest.get_label(), test_pred_y, title='black_basemobile_ROC-line')
