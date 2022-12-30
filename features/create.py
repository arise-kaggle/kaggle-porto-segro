import pandas as pd
import numpy as np
import re as re
from sklearn.preprocessing import OneHotEncoder


from base import Feature, get_arguments, generate_features

Feature.dir = "features"


class Base_features(Feature):
    def create_features(self):
        base_columns = [
            "ps_ind_01",
            "ps_ind_02_cat",
            "ps_ind_03",
            "ps_ind_04_cat",
            "ps_ind_05_cat",
            "ps_ind_06_bin",
            "ps_ind_07_bin",
            "ps_ind_08_bin",
            "ps_ind_09_bin",
            "ps_ind_10_bin",
            "ps_ind_11_bin",
            "ps_ind_12_bin",
            "ps_ind_13_bin",
            "ps_ind_14",
            "ps_ind_15",
            "ps_ind_16_bin",
            "ps_ind_17_bin",
            "ps_ind_18_bin",
            "ps_reg_01",
            "ps_reg_02",
            "ps_reg_03",
            "ps_car_01_cat",
            "ps_car_02_cat",
            "ps_car_03_cat",
            "ps_car_04_cat",
            "ps_car_05_cat",
            "ps_car_06_cat",
            "ps_car_07_cat",
            "ps_car_08_cat",
            "ps_car_09_cat",
            "ps_car_10_cat",
            "ps_car_11_cat",
            "ps_car_11",
            "ps_car_12",
            "ps_car_13",
            "ps_car_14",
            "ps_car_15",
            "ps_calc_01",
            "ps_calc_02",
            "ps_calc_03",
            "ps_calc_04",
            "ps_calc_05",
            "ps_calc_06",
            "ps_calc_07",
            "ps_calc_08",
            "ps_calc_09",
            "ps_calc_10",
            "ps_calc_11",
            "ps_calc_12",
            "ps_calc_13",
            "ps_calc_14",
            "ps_calc_15_bin",
            "ps_calc_16_bin",
            "ps_calc_17_bin",
            "ps_calc_18_bin",
            "ps_calc_19_bin",
            "ps_calc_20_bin",
        ]
        self.train[base_columns] = train[base_columns]
        self.test[base_columns] = test[base_columns]


class Num_missing(Feature):
    def create_features(self):
        self.train["num_missing"] = (train == -1).sum(axis=1)
        self.test["num_missing"] = (test == -1).sum(axis=1)

class Count_cat_features(Feature):
    def create_features(self):
        
        ind_features = [col for col in train.columns if 'ind' in col]

        first_col=True
        for col in ind_features:
            if first_col:
                train['mix_ind'] = train[col].astype(str)+'_'
                test['mix_ind'] = test[col].astype(str)+'_'
                first_col = False
            else:
                train['mix_ind'] += train[col].astype(str)+'_'
                test['mix_ind'] += test[col].astype(str)+'_'

        all_data = train.append(test).copy()
        cat_features = [col for col in train.columns if 'cat' in col] 
        for col in cat_features+['mix_ind']:
            val_counts_dic = all_data[col].value_counts().to_dict()
            self.train[f'{col}_count'] = train[col].apply(lambda x: val_counts_dic[x])
            self.test[f'{col}_count'] = test[col].apply(lambda x: val_counts_dic[x])


if __name__ == "__main__":
    args = get_arguments()

    train = pd.read_feather("./data/input/train.feather")
    test = pd.read_feather("./data/input/test.feather")

    generate_features(globals(), args.force)
