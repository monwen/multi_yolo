import pandas as pd
import numpy as np
import accuracy_ex_torch

def load_test():
    acc = accuracy_ex_torch.ACCU()
    acc.load('base_608_txt')
    print(acc.df)

if __name__ == '__main__':
    load_test()
