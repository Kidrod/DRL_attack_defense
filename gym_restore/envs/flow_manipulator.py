import re
import copy
import numpy as np
import pandas as pd

class flow_Manipulator(object):
    def __init__(self):
        pass

    ####################random#######################
    # ACTION_TABLE = {
    # 'delete_obj': 'delete_obj',
    # 'delete_stream': 'delete_stream',
    # "delete_page":"delete_page",
    # "delete_xref":"delete_xref",
    # "delete_trailer":"delete_trailer"
    # }
    #
    # def delete_obj(self,sample):
    #     sample[0][0] -= 1
    #     sample[0][1] -= 1
    #     return sample
    #
    # def delete_stream(self,sample):
    #     sample[0][2] -= 1
    #     sample[0][3] -= 1
    #     return sample
    #
    # def delete_page(self,sample):
    #     sample[0][7]  -= 1
    #     return sample
    #
    # def delete_xref(self,sample):
    #     sample[0][4] -= 1
    #     return sample
    #
    # def delete_trailer(self,sample):
    #     sample[0][5] -= 1
    #     return sample



    ####################feature analysis#######################
    ACTION_TABLE = {
    'add_js': 'add_js',
    'delete_js': 'delete_js',
    "add_page":"add_page",
    "delete_page":"delete_page",
    "add_obj":"add_obj",
    "delete_obj":"delete_obj",
    }

    def add_js(self,sample):
        sample[0][10] += 1
        sample[0][11] += 1
        return sample

    def delete_js(self,sample):
        sample[0][10] -= 1
        sample[0][11] -= 1
        return sample

    def add_page(self,sample):
        sample[0][7] += 1
        return sample

    def delete_page(self,sample):
        sample[0][7]  -= 1
        return sample

    def add_obj(self,sample):
        sample[0][0] += 1
        sample[0][1] += 1
        return sample

    def delete_obj(self,sample):
        sample[0][0] -= 1
        sample[0][1] -= 1
        return sample

    def modify(self,sample, _action):
        origin = copy.deepcopy(sample)
        action_func=flow_Manipulator().__getattribute__(_action)
        result = action_func(origin)
        return result  if (np.array(result[0]) >= 0).all() else sample

if __name__ == '__main__':
    f=flow_Manipulator()
    df = pd.read_csv(r"../../data_for_defense/evaded_samples_with_origin/evaded_for_attack.csv", index_col=None)
    X = df["features"].apply(lambda x: re.findall(r"\d+\.?\d*", x)).values.tolist()
    X = [list(map(lambda x: int(x), i)) for i in X]

    k = [X[-2]]
    print(k)
    for i in range(10):
        k=f.modify(k,"delete_xref")
        print(k)
