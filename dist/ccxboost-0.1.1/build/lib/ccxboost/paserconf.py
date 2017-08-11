import configparser
from sklearn.grid_search import ParameterGrid

'''解析conf配置文件'''


def extract_conf(conf_path, conf_section):
    conf = configparser.ConfigParser()
    conf.read(conf_path)
    kvs = conf.items(conf_section)

    param = {}
    for (m, n) in kvs:
        n_v = n.split(',')
        new_n_v = []
        for j in n_v:
            try:
                try:
                    new_n_v.append(int(j))
                except:
                    new_n_v.append(float(j))
            except:
                new_n_v.append(j)
        param[m] = new_n_v
    return param


# param = extract_conf(r"C:\Users\liyin\Desktop\ccxboost\ccxboost\ccxboost.conf", 'XGB_PARAMS')
# params = list(ParameterGrid(param))
# len(params)
