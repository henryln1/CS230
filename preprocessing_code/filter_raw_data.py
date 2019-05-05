import pandas as pd
import sys
import lzma
import json

'''
Process raw data files from pushshift.io and extract only what we need.
'''

subreddits = ['AskReddit']

submissions_list = []

def process_file(file_name):
	with lzma.open(file_name) as file:
		counter = 0
		for line in file:
			json_format_line = json.loads(line)
			subreddit = json_format_line['subreddit']
			if subreddit in subreddits:
				score = json_format_line['score']
				title = json_format_line['title']
				title = title.replace(',', '') #should replace with a whole preprocessing function later
				number_comments = json_format_line['num_comments']
				timestamp = json_format_line['created_utc']
				submissions_list.append((subreddit, title, score, number_comments, timestamp)) #append this to list
			counter += 1
			if counter % 100000 == 0:
				print(counter, " lines processed.")

def write_to_csv(file_name):
	print("Writing to CSV located at " + file_name)
	column_names = ['Subreddit', 'Title', 'Score', 'Num_Comments', 'Timestampe']
	df = pd.DataFrame(submissions_list, columns = column_names)
	df.to_csv(file_name)


def main():
	print("Beginning to process .xz file..")
	file_directory = '../../project_data/'
	file_name = 'RS_2018-09.xz'
	csv_directory = file_directory + 'csv_files/'
	csv_file_name = csv_directory + file_name[:-3] + '_' + subreddits[0] + '_submissions.csv'
	print("Csv file name: ", csv_file_name)
	process_file(file_directory + file_name)
	write_to_csv(csv_file_name)
	print("Done writing to csv file. Exiting.")






if __name__ == "__main__":
	main()