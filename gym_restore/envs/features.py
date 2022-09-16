#-*- coding:utf-8 –*-
import numpy as np

class Restore_Features(object):
    def __init__(self):
        pass

    def feature_extraction(self,filepath):
        k = np.asarray(filepath[0])
        return k

if __name__ == '__main__':
    f=Restore_Features()
    k = r"D:\Download\DRL_attack_defense\malicious_train_copy\1.pdf"
    a=f.feature_extraction(k)
    print(a)