import os
import shutil
import pandas as pd
import pdfplumber
import pdfrw
from pdfrw import PdfReader

def get_filelist(dir, Filelist):
    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)
    return Filelist

if __name__ == '__main__':
    # df = pd.read_csv(r"D:\Download\DRL_attack_defense\no_text.csv",index_col=None)

    list = get_filelist(r'D:\Download\DRL_attack_defense\malicious_for_evade', [])
    i = 0
    for e in list:
        if i < 2826:
            shutil.copy(e, r"D:\Download\DRL_attack_defense\malicious_train")
        else:
            shutil.copy(e, r"D:\Download\DRL_attack_defense\malicious_test")
        i += 1

        ##############compatible with PdfReader, then move to malicious_final
        # os.rename(e,r"D:\Download\DRL_attack_defense\malicious_all\{}.pdf".format(e[e.rfind('\\') + 1:]))

        # try:
        #     if PdfReader(e):
        #         print("yes")
        #         shutil.move(e, r"D:\Download\DRL_attack_defense\malicious_for_evade")
        # except pdfrw.errors.PdfParseError:
        #     continue
        # except ValueError:
        #     continue
        # except StopIteration:
        #     continue

        # result = df[(df["filename"] == e[e.rfind('\\') + 1:].split(".")[0])]
        # if result.empty:
        #     pass
        # else:
        #     # name_pre = e[e.rfind('\\') + 1:]
        #     shutil.move(e, r"D:\Download\DRL_attack_defense\malicious_final")
        #     # os.rename(r"D:\Download\DRL_attack_defense\malicious_all\{}".format(name_pre),
        #     #           r"D:\Download\DRL_attack_defense\malicious_all\{}.pdf".format(name_pre))
