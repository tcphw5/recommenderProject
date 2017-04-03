# -*- coding: utf-8 -*-
"""
Reccomender Systems Project first file

Created on Wed Mar 22 11:28:54 2017

@author: tpwin10
"""

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import NMF
from surprise import evaluate, print_perf
from surprise import KNNBasic
import os

#load data from file
fileIN = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(fileIN, reader=reader)

data.split(n_folds=3)

"""
parts 5-12 of HW4.pdf

"""
algs = [SVD(), SVD(biased=False), NMF(), KNNBasic(sim_options={'user_based': True}), KNNBasic(sim_options={'user_based': False})]

fold1 = {'mae':[], 'rmse':[]}
fold2 = {'mae':[], 'rmse':[]}
fold3 = {'mae':[], 'rmse':[]}
means = {'mae':[], 'rmse':[]}

for alg in algs:
    perf = evaluate(alg, data, measures=['RMSE', 'MAE'])
    
    fold1['mae'].append(perf['mae'][0])
    fold2['mae'].append(perf['mae'][1])
    fold3['mae'].append(perf['mae'][2])
    means['mae'].append(sum(perf['mae'])/len(perf['mae']))
    fold1['rmse'].append(perf['rmse'][0])
    fold2['rmse'].append(perf['rmse'][1])
    fold3['rmse'].append(perf['rmse'][2])
    means['rmse'].append(sum(perf['rmse'])/len(perf['rmse']))
    
    print_perf(perf)

'''
part 13 of HW4.pdf

"""
CFalgs = [KNNBasic(sim_options={'user_based': True, 'name':'MSD'}), KNNBasic(sim_options={'user_based': True,'name':'cosine'}), KNNBasic(sim_options={'user_based': True,'name':'pearson'}), KNNBasic(sim_options={'user_based': False,'name':'MSD'}), KNNBasic(sim_options={'user_based': False, 'name':'cosine'}), KNNBasic(sim_options={'user_based': False,'name':'pearson'})]

fold1cf = {'mae':[], 'rmse':[]}
fold2cf = {'mae':[], 'rmse':[]}
fold3cf = {'mae':[], 'rmse':[]}
meanscf = {'mae':[], 'rmse':[]}

for alg in CFalgs:
    perf = evaluate(alg, data, measures=['RMSE', 'MAE'])
    
    fold1cf['mae'].append(perf['mae'][0])
    fold2cf['mae'].append(perf['mae'][1])
    fold3cf['mae'].append(perf['mae'][2])
    meanscf['mae'].append(sum(perf['mae'])/len(perf['mae']))
    fold1cf['rmse'].append(perf['rmse'][0])
    fold2cf['rmse'].append(perf['rmse'][1])
    fold3cf['rmse'].append(perf['rmse'][2])
    meanscf['rmse'].append(sum(perf['rmse'])/len(perf['rmse']))
    
    print_perf(perf)


"""
part 14 of HW4.pdf

'''

'''
#CFKmodalgs = [KNNBasic(k=10, sim_options={'user_based': True, 'name':'MSD'}), KNNBasic(k=20, sim_options={'user_based': True,'name':'MSD'}), KNNBasic(k=30, sim_options={'user_based': True,'name':'MSD'}), KNNBasic(k=10, sim_options={'user_based': False,'name':'MSD'}), KNNBasic(k=20, sim_options={'user_based': False, 'name':'MSD'}), KNNBasic(k=30, sim_options={'user_based': False,'name':'MSD'})]

for kval in range(5,55,5):
    currentUBALG = KNNBasic(k=kval, sim_options={'user_based': True, 'name':'MSD'})
    currentIBALG = KNNBasic(k=kval, sim_options={'user_based': False, 'name':'MSD'})
    

    perf = evaluate(currentIBALG, data, measures=['RMSE'])
    perf2 = evaluate(currentUBALG, data, measures=['RMSE'])
    
    
    print_perf(perf)
    print_perf(perf2)
'''