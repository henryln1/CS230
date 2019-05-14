import random
import numpy as np
np.random.seed(7)

import csv
import pandas as pd



#files for each data split

TRAIN_FILE = 'train_data.csv'
DEV_FILE = 'dev_data.csv'
TEST_FILE = 'test_data.csv'

#proportion

TRAIN_PERCENTAGE = 0.7
DEV_PERCENTAGE = (1 - TRAIN_PERCENTAGE) / 2
TEST_PERCENTAGE = (1 - TRAIN_PERCENTAGE) / 2


COLUMN_NAMES = ['Subreddit', 'Title', 'Score', 'Num_Comments', 'Timestampe']



def write_to_files(train, dev, test):
	df = pd.DataFrame(train, columns = COLUMN_NAMES)
	df.to_csv(TRAIN_FILE)
	df = pd.DataFrame(dev, columns = COLUMN_NAMES)
	df.to_csv(DEV_FILE)
	df = pd.DataFrame(test, columns = COLUMN_NAMES)
	df.to_csv(TEST_FILE)	

def split_file(file_name):
	'''
	Takes a (shuffled) CSV file, splits it 
	and then outputs it into 3 new csv
	'''
	num_data_points = sum(1 for line in open(file_name)) - 1
	train_data_points = int(TRAIN_PERCENTAGE * num_data_points)
	dev_data_points = (num_data_points - train_data_points) // 2
	test_data_points = (num_data_points - train_data_points) // 2

	train_rows_range = (1, train_data_points)
	dev_rows_range = (train_data_points, train_data_points + dev_data_points)
	test_rows_range = (train_data_points + dev_data_points, train_data_points + dev_data_points + test_data_points)
	print("Train Range: ", train_rows_range)
	print("Dev Range:" , dev_rows_range)
	print("Test Range: ", test_rows_range)
	with open(file_name) as fd_1:
		reader = csv.reader(fd_1)
		train_data = [row[1:] for idx, row in enumerate(reader) if idx > train_rows_range[0] and idx < train_rows_range[1]]
	fd_1.close()
	with open(file_name) as fd_2:
		reader = csv.reader(fd_2)
		dev_data = [row[1:] for idx, row in enumerate(reader) if idx > dev_rows_range[0] and idx < dev_rows_range[1]]
	fd_2.close()
	with open(file_name) as fd_3:	
		reader = csv.reader(fd_3)
		test_data = [row[1:] for idx, row in enumerate(reader) if idx > test_rows_range[0] and idx < test_rows_range[1]]
	fd_3.close()
	print("Training Data Size: ", len(train_data))
	print("Dev Data Size: ", len(dev_data))
	print("Test Data Size: ", len(test_data))
	print("First Training Sample: ", train_data[0])
	print("First Dev Sample: ", dev_data[0])
	print("First Test Sample: ", test_data[0])
	write_to_files(train_data, dev_data, test_data)

def shuffle_file(file_name):
	#shuffles a csv and outputs it to the same csv
	data_points = []
	with open(file_name) as fd:
		reader = csv.reader(fd)
		data_points = [row[1:] for idx, row in enumerate(reader) if idx > 0]
	np.random.shuffle(data_points)
	df = pd.DataFrame(data_points, columns = COLUMN_NAMES)
	df.to_csv(file_name)

def main():
	file_name = 'preprocessing_code/RS_2018-09_AskReddit_submissions_commas_removed.csv'
	shuffle_file(file_name)
	split_file(file_name)

if __name__ == "__main__":
	main()

