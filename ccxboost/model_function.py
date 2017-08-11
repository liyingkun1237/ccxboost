'''

建模处理各个过程封装的函数
@author:liyingkun

'''
import configparser
import time
import xgboost as xgb
import logging
import datetime
import os
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.switch_backend('agg')  # 解决matplotlib在Linux下图片不能显示的报错问题

conf = configparser.ConfigParser()
conf_path = os.path.split(os.path.realpath(__file__))[0] + '/ccxboost.conf'
conf.read(conf_path)
root_path = conf.get("DIRECTORY", "project_pt")


# 0802新增，为了适应将结果写到不同的时间戳文件夹下
def model_result_path(root_path):
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    filename = 'model' + timestamp
    path = root_path + filename
    if os.path.exists(path):
        return path
    else:
        os.mkdir(path)
        return path


root_path = model_result_path(root_path)


# 0803添加，将结果路径保存到指定的文件夹下

def model_data(df, x_colnames, y_colnames, miss=np.nan):
    # 数据准备
    ddf = xgb.DMatrix(df[x_colnames], label=df[y_colnames], missing=miss)
    return ddf


# 后续改进，从配置文件获取出日志的存储路径
def model_infologger(message):
    path = root_path + '/modellog'
    if os.path.exists(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)

    format = '%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s:\n %(message)s'
    curDate = datetime.date.today() - datetime.timedelta(days=0)
    infoLogName = r'%s_info_%s.log' % (message, curDate)

    formatter = logging.Formatter(format)

    infoLogger = logging.getLogger('%s_info_%s.log' % (message, curDate))

    #  这里进行判断，如果logger.handlers列表为空，则添加，否则，直接去写日志
    if not infoLogger.handlers:
        infoLogger.setLevel(logging.INFO)

        infoHandler = logging.FileHandler(infoLogName, 'a')
        infoHandler.setLevel(logging.INFO)
        infoHandler.setFormatter(formatter)
        infoLogger.addHandler(infoHandler)

    os.chdir(root_path)

    return infoLogger


'''交叉验证+网格搜索'''


def model_cv(dtrain, param_grid, num_boost_round, nfold=5, message='cv_allcol_1'):
    # 初始化日志文件
    message = 'model_cv_' + message
    infoLogger = model_infologger(message)

    ''' 定制化评价函数'''

    def kserro(preds, dtrain):
        fpr, tpr, thresholds = roc_curve(dtrain.get_label(), preds)
        ks = np.max(tpr - fpr)
        return 'ks', ks

    params = list(ParameterGrid(param_grid))
    i = 0
    result = []
    for param in params:
        re = xgb.cv(param, dtrain, num_boost_round, nfold=nfold, verbose_eval=True)
        param['max_test_auc'] = re.iloc[np.argmax(re['test-auc-mean'])]['test-auc-mean']
        param['max_train_auc'] = re.iloc[np.argmax(re['test-auc-mean'])]['train-auc-mean']
        # 0624 ks的加入，特别说明，auc和ks各自取最大 feval=kserro,
        # param['max_test_ks'] = re.iloc[np.argmax(re['test-ks-mean'])]['test-ks-mean']
        # param['max_train_ks'] = re.iloc[np.argmax(re['test-ks-mean'])]['train-ks-mean']

        param['num_best_round'] = re['test-auc-mean'].diff().abs().argmin() + 1  # 一阶差分不会增加，迭代次数不能带来实质的提高
        param['num_maxtest_round'] = np.argmax(re['test-auc-mean']) + 1

        param['num_boost_round'] = num_boost_round
        result.append(param)

        infoLogger.info(param)
        infoLogger.info('max_test_auc:%s' % param['max_test_auc'])
        infoLogger.info('num_maxtest_round:%d' % param['num_maxtest_round'])
        infoLogger.info('max_train_auc:%s' % param['max_train_auc'])
        infoLogger.info('num_best_round:%d' % param['num_best_round'])

        i += 1
        infoLogger.info('<<<总计选参：%d,已运行到：第 %d 个参数>>>\n\n' % (len(params), i))

        print('\n\n<<<总计选参：%d,已运行到：%d 个参数>>>\n\n' % (len(params), i))

        # save_data(pd.DataFrame(result), message + '.csv')
        pd.DataFrame(result).to_csv((message + '.csv'))

    result = pd.DataFrame(result)

    return result


def get_bstpram(re, method='defualt'):
    if np.argmax(re['max_test_auc']) == np.argmin(re['max_train_auc'] - re['max_test_auc']):
        print('warning:two method have eqaul bst result.')

    if method == 'defualt':
        ipos = np.argmax((re['max_train_auc'] + re['max_test_auc'])
                         / np.round(re['max_train_auc'] - re['max_test_auc'], 3))
        param = dict(re.iloc[ipos, :])
        num_round = param['num_maxtest_round']
        return param, num_round


def model_train(dtrain_text, dtest_text, param, num_round):
    '''
    模型训练
    '''

    def kserro(preds, dtrain):
        fpr, tpr, thresholds = roc_curve(dtrain.get_label(), preds)
        ks = np.max(tpr - fpr)
        return 'ks', ks

    watchlist = [(dtest_text, 'test'), (dtrain_text, 'train')]
    bst = xgb.train(param, dtrain_text, num_round, watchlist, verbose_eval=True)
    # feval=kserro,
    return bst


def save_bstmodel(bst, mess):
    path = root_path + '/modeltxt'
    if os.path.exists(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)

    curDate = datetime.date.today() - datetime.timedelta(days=0)
    path_1 = 'model_' + mess + '_' + str(curDate) + '.txt'
    with open(path_1, 'wb') as f:
        pickle.dump(bst, f)
    os.chdir(root_path)
    print('模型保存成功 文件路径名：%s' % (path + '/' + path_1))
    return path + '/' + path_1


def load_bstmodel(path):
    with open(path, 'rb') as f:
        bst = pickle.load(f)
    return bst


def get_importance_var(bst):
    '''
    获取进入模型的重要变量
    '''
    re = pd.Series(bst.get_score(importance_type='gain')).sort_values(ascending=False)
    re = pd.DataFrame(re, columns=['value']).reset_index()
    re.columns = ['var_Name', 'value']
    re = re.assign(
        pct_importance=lambda x: x['value'].apply(lambda s: str(np.round(s / np.sum(x['value']) * 100, 2)) + '%'))
    print('重要变量的个数：%d' % len(re))
    return re


def ks(y_pred, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks = np.max(tpr - fpr)
    return ks


def AUC(y_pred, y_true):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    return roc_auc


def model_predict(bst, dtrain_text, dtest_text, message='data_id'):
    train_pred_y_xg = bst.predict(dtrain_text)
    test_pred_y_xg = bst.predict(dtest_text)

    train_report = classification_report(dtrain_text.get_label(), train_pred_y_xg > 0.5)
    test_report = classification_report(dtest_text.get_label(), test_pred_y_xg > 0.5)
    print('训练集模型报告：\n', train_report)
    print('测试集模型报告：\n', test_report)

    # 初始化日志文件，保存模型结果
    message = 'model_report_' + str(message)
    infoLogger = model_infologger(message)
    infoLogger.info('train_report:\n%s' % train_report)
    infoLogger.info('test_report:\n%s' % test_report)

    ks_train = ks(train_pred_y_xg, dtrain_text.get_label())

    ks_test = ks(test_pred_y_xg, dtest_text.get_label())

    print('ks_train: %f,ks_test：%f' % (ks_train, ks_test))
    infoLogger.info('ks_train: %f,ks_test：%f \n\n' % (ks_train, ks_test))

    return train_pred_y_xg, test_pred_y_xg


def get_modelpredict_re(test_index, test_pred):
    re = pd.DataFrame([test_index, test_pred]).T
    re.rename(columns={'Unnamed 0': 'P_value'}, inplace=True)
    return re


def plot_imp(bst):
    ax = xgb.plot_importance(bst, grid=False)
    # fig = plt.gca()
    # save_figure(fig, 'Feature importance')
    # fig.show()
    return ax


def plot_ks_line(y_true, y_pred, title='ks-line', detail=False):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    plt.plot(tpr, label='tpr-line')
    plt.plot(fpr, label='fpr-line')
    plt.plot(tpr - fpr, label='KS-line')
    # 设置x的坐标轴为0-1范围
    plt.xticks(np.arange(0, len(tpr), len(tpr) // 10), np.arange(0, 1.1, 0.1))

    # 添加标注
    x0 = np.argmax(tpr - fpr)
    y0 = np.max(tpr - fpr)
    plt.scatter(x0, y0, color='black')  # 显示一个点
    z0 = thresholds[x0]  # ks值对应的阈值
    plt.text(x0 - 2, y0 - 0.12, ('(ks: %.4f,\n th: %.4f)' % (y0, z0)))

    if detail:
        # plt.plot([x0,x0],[0,y0],'b--',label=('thresholds=%.4f'%z0)) #在点到x轴画出垂直线
        # plt.plot([0,x0],[y0,y0],'r--',label=('ks=%.4f'%y0)) #在点到y轴画出垂直线
        plt.plot(thresholds[1:], label='thresholds')
        t0 = thresholds[np.argmin(np.abs(thresholds - 0.5))]
        t1 = list(thresholds).index(t0)
        plt.scatter(t1, t0, color='black')
        plt.plot([t1, t1], [0, t0])
        plt.text(t1 + 2, t0, 'thresholds≈0.5')

        tpr0 = tpr[t1]
        plt.scatter(t1, tpr0, color='black')
        plt.text(t1 + 2, tpr0, ('tpr=%.4f' % tpr0))

        fpr0 = fpr[t1]
        plt.scatter(t1, fpr0, color='black')
        plt.text(t1 + 2, fpr0, ('fpr=%.4f' % fpr0))

    plt.legend(loc='upper left')
    plt.title(title)
    fig_path = save_figure(plt, title)
    plt.show()
    plt.close()
    return fig_path


'''
封装一个函数：绘制ROC曲线
'''


def plot_roc_line(y_true, y_pred, title='ROC-line'):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    ks = np.max(tpr - fpr)
    plt.plot(fpr, tpr)  # ,label=('auc= %.4f'%roc_auc)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    plt.text(0.7, 0.45, ('auc= %.4f \nks  = %.4f' % (roc_auc, ks)))

    plt.title(title)
    fig_path = save_figure(plt, title)
    plt.show()
    plt.close()
    return fig_path


def load_data(path, *args):
    data = pd.read_csv(path, *args)
    return data


def save_data(data, data_name, index=False):
    curDate = datetime.date.today() - datetime.timedelta(days=0)
    path = root_path + '/modeldata'
    if os.path.exists(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)

    data_name = 'd_' + str(curDate) + '_' + data_name
    data.to_csv(data_name, index=index)

    os.chdir(root_path)
    print('数据保存成功:%s' % (path + '/' + data_name))
    return path + '/' + data_name


def save_figure(fig, fig_name):
    curDate = datetime.date.today() - datetime.timedelta(days=0)
    path = root_path + '/modelfig'
    if os.path.exists(path):
        os.chdir(path)
    else:
        os.mkdir(path)
        os.chdir(path)

    fig_name = 'd_' + str(curDate) + '_' + fig_name
    fig.savefig(fig_name)
    print('图片保存成功:%s' % (path + '/' + fig_name + '.png'))
    os.chdir(root_path)
    return path + '/' + fig_name + '.png'


def my_splitdata_f1(data, x_columns, y_columns, testsize=0.3, rdm_state=701):
    X = data[x_columns]
    Y = data[y_columns]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testsize,
                                                        random_state=rdm_state, stratify=Y)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    return train, test

def write_path(file, path_list):
    with open(file, 'w') as f:
        f.writelines([line + '\n' for line in path_list])
        f.write('\n')
    print('结果路径写入到%s文件中' % file)




