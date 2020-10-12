# -*- coding: utf-8 -*-
# @Time : 2020/10/10 2:31 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : pst.py


import pandas as pd
import numpy as np
from toad import detect
from toad import quality
from pathlib import Path
from utils import check_is_dir, rescaling
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
            self._detect_data = detect(self._target_data)
            self._vars_type_mapping = dict(zip(self._detect_data.index.to_list(),
                                               self._detect_data.type.astype(str).to_list()))
            # print(self._vars_type_mapping)
        return self._detect_data

    @property
    def univariate_analysis(self) -> str:
        # 1. plot the vars distribute
        # 2. plot the vars distribute group by target
        # 3. plot the ar with target
        for var in self._vars_type_mapping.keys():
            if var != self._taget:
                if any(map(lambda x: x in self._vars_type_mapping[var], ['int', 'float'])):
                    self._plot_distribute[var] = hist_distribute(self._target_data[var], var)

                    self._plot_distribute_target[var] = hist_distribute_with_target(
                        self._target_data[[var, self._taget]].copy(), var, self._taget)
                    if var == 'Q1':
                        self._plot_ar[var] = cap_plot(rescaling(self._target_data[var]), self._target_data[self._taget])

                else:
                    self._plot_distribute[var] = hist_distribute(self._target_data[var], var)
                    self._plot_distribute_target[var] = hist_distribute_with_target_norminal(
                        self._target_data[[var, self._taget]].copy(), var, self._taget)

        self._flag_univariate_analysis = True

        return r'univariate analysis finished， you can call distribute to get plots'

    @property
    def distribute(self):
        if not self._flag_univariate_analysis:
            raise PermissionError("distribute will be return when univariate_analysis finished")
        return self._plot_distribute

    def save_figure(self, save_dir_path):
        if not check_is_dir(save_dir_path):
            raise IsADirectoryError(r'please give a correct dir_path')
        save_dir_path = Path(save_dir_path)

        if not self._flag_univariate_analysis:
            raise PermissionError("save_figure will be excute when univariate_analysis finished")

        for name, fig in self._plot_distribute.items():
            fig.savefig(save_dir_path / '{0}_{1}.png'.format(name, 'distribute'))

        for name, fig in self._plot_distribute_target.items():
            fig.savefig(save_dir_path / '{0}_{1}.png'.format(name, 'distribute_taget'))

        for name, fig in self._plot_ar.items():
            fig.savefig(save_dir_path / '{0}_{1}.png'.format(name, 'ar'))

        return 'figure save finished, please check it'


if __name__ == '__main__':
    data = pd.read_csv(r'/Users/quanbing/Downloads/workspace/competition/binary classification/交通事故理赔审核/data/'
                       r'train.csv', index_col=[0])
    import random
    data['性别'] = np.array([random.choice(['男', '女', '女', '女', '跨性别']) for i in range(data.shape[0])])
    bx = Pst(data, 'Evaluation')
    bx.desc
    bx.univariate_analysis
    bx.save_figure(r'/Users/quanbing/Downloads/code/tmp')



