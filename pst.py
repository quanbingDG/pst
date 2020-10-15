# -*- coding: utf-8 -*-
# @Time : 2020/10/10 2:31 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : pst.py


import pandas as pd
import numpy as np
import toad
from tqdm import tqdm
from pathlib import Path
from pst_metrics import mutual_info_matrix
from utils import check_is_dir, rescaling, save_fig
from plot import hist_distribute, hist_distribute_with_target, hist_distribute_with_target_norminal, cap_plot


class Pst:
    '''
    analysis class
    '''
    def __init__(self, ori_data: pd.DataFrame, target: str) -> None:
        '''
        :param ori_data: a pandas dataframe
        '''
        self._ori_data = ori_data
        self._target_data = ori_data.copy()
        self._detect_data = None
        self._quality_data = None
        self._plot_distribute = {}
        self._plot_distribute_target = {}
        self._plot_ar = {}
        self._vars_type_mapping = None
        self._taget = target
        self._model_columns = ori_data.columns.to_list()
        self._corr_matrix = {}
        self._combiner = toad.transform.Combiner()
        self._woe_transer = toad.transform.WOETransformer()
        self._flag_corr_matrix = False
        self._flag_univariate_analysis = False

    @property
    def ori_data(self):
        return self._ori_data

    @ori_data.setter
    def ori_data(self, value: pd.DataFrame):
        if self._ori_data is None:
            self._ori_data = value
            self._target_data = value.copy()
        else:
            raise PermissionError("ori_data just be Assignment when it's None")

    @property
    def target_data(self):
        return self._target_data

    @target_data.setter
    def target_data(self, value):
        self._target_data = value

    @property
    def desc(self):
        if self._detect_data is None:
            self._detect_data = toad.detect(self._target_data)
            self._vars_type_mapping = dict(zip(self._detect_data.index.to_list(),
                                               self._detect_data.type.astype(str).to_list()))
            # print(self._vars_type_mapping)
        return self._detect_data

    @property
    def distribute(self):
        if not self._flag_univariate_analysis:
            raise PermissionError("distribute will be return when univariate_analysis finished")
        return self._plot_distribute

    @property
    def corr_matrix(self):
        if not self._flag_corr_matrix:
            self._corr_matrix['pearson'] = self._target_data.corr(method='pearson')
            self._corr_matrix['kendall'] = self._target_data.corr(method='kendall')
            self._corr_matrix['spearman'] = self._target_data.corr(method='spearman')
            self._corr_matrix['mutual_info'] = mutual_info_matrix(self._target_data)
            self._flag_corr_matrix = True

        return self._corr_matrix

    @property
    def data_quality(self):
        if not self._quality_data:
            self._quality_data = toad.quality(self._target_data, target=self._target_data)
        return self._quality_data

    def univariate_analysis(self, ar=False) -> str:
        # 1. plot the vars distribute
        # 2. plot the vars distribute group by target
        # 3. plot the ar with target
        p_bar = tqdm(self._vars_type_mapping.keys())
        for var in p_bar:
            p_bar.set_description("Processing univariate analysis %s" % var)
            if var != self._taget:
                if any(map(lambda x: x in self._vars_type_mapping[var], ['int', 'float'])):
                    self._plot_distribute[var] = hist_distribute(self._target_data[var], var)

                    self._plot_distribute_target[var] = hist_distribute_with_target(
                        self._target_data[[var, self._taget]].copy(), var, self._taget)
                    if ar:
                        self._plot_ar[var] = cap_plot(rescaling(self._target_data[var]),
                                                      self._target_data[self._taget])
                else:
                    self._plot_distribute[var] = hist_distribute(self._target_data[var], var)

                    self._plot_distribute_target[var] = hist_distribute_with_target_norminal(
                        self._target_data[[var, self._taget]].copy(), var, self._taget)

        self._flag_univariate_analysis = True

    def save_figure(self, save_dir_path):
        if not check_is_dir(save_dir_path):
            raise IsADirectoryError(r'please give a correct dir_path')
        save_dir_path = Path(save_dir_path)

        if not self._flag_univariate_analysis:
            raise PermissionError("save_figure will be excute when univariate_analysis finished")

        save_fig(self._plot_distribute, save_dir_path, type_str='distribute')
        save_fig(self._plot_distribute_target, save_dir_path, type_str='distribute_taget')
        save_fig(self._plot_ar, save_dir_path, type_str='ar')

    def feature_select(self, empty: float = 0.9, iv: float = 0.02, corr: float = 0.7, return_drop: bool = False,
                       exclude=None):

        re_select = toad.select(self._target_data, empty=empty, iv=iv, corr=corr, return_drop=return_drop,
                                exclude=exclude)

        if not return_drop:
            self._model_columns = re_select.columns.to_list()
        self._model_columns = re_select[0].columns.to_list()


if __name__ == '__main__':
    data = pd.read_csv(r'/Users/quanbing/Downloads/workspace/competition/binary classification/交通事故理赔审核/data/'
                       r'train.csv', index_col=[0])
    import random
    data['性别'] = np.array([random.choice(['男', '女', '女', '女', '跨性别']) for i in range(data.shape[0])])
    bx = Pst(data, 'Evaluation')
    bx.desc
    # print(bx.corr_matrix)
    bx.univariate_analysis()
    bx.save_figure(r'/Users/quanbing/Downloads/code/tmp')



