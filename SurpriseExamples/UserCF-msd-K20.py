from surprise import KNNBasic
from surprise import Dataset
from surprise import evaluate, print_perf
from surprise import Reader
import os

#load data from a file 
file_path = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

data.split(n_folds=5)
algo = KNNBasic(k=20, sim_options = {
               'user_based': True 
               })
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
print_perf(perf)
