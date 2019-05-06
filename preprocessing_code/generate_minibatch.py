import sys
import numpy as np
import random
import re
import csv

EXAMPLE_LENGTH = 100






def build_minibatch(minibatch_size, glove_vectors, data_file_location):
	glove_vector_dim = len(glove_vectors['the'])
	X = np.zeros(shape = (minibatch_size, EXAMPLE_LENGTH, glove_vector_dim))
	Y = np.zeros(shape = (minibatch_size, 1))
	num_data_points = num_lines = sum(1 for line in open(data_file_location))
	random_rows = [random.randint(1, num_data_points) for x in range(minibatch_size)]
	data_points = []
	with open(data_file_location) as fd:
		reader = csv.reader(fd)
		data_points = [row for idx, row in enumerate(reader) if idx in random_rows]
	print(data_points[0])
	for i in range(len(data_points)):
		_, subreddit, title, score, num_comments, timestamp = tuple(data_points[i])
		print("Title: ", title)
		title_words_list = re.sub(r'[^a-zA-Z ]', '', title).split()
		title_words_list = [x.lower() for x in title_words_list]
		print("Words: ", title_words_list)
		for j in range(len(title_words_list)):
			curr_word = title_words_list[j]
			if curr_word in glove_vectors:
				X[i][j] = glove_vectors[curr_word]
		print("X: ", X[i])
		Y[i] = score
		print("Y: ", Y[i])
	return X, Y



def build_glove_dict(file_name):
	glove_dict = {}
	with open(file_name, 'r') as f:
		for line in f:
			line_list = line.split(" ")
			word = line_list[0]
			embeddings = line_list[1:]
			assert(str(len(embeddings)) in file_name) #making sure dimensions match file name
			embeddings = [float(x) for x in embeddings]
			glove_dict[word] = embeddings
	f.close()
	return glove_dict


def main():
	assert len(sys.argv) == 3 #make sure we are given a glove vector dim and minibatch size
	glove_vector_dimensions = sys.argv[1]
	minibatch_size = int(sys.argv[2])
	data_file_location = '../../project_data/csv_files/RS_2018-09_AskReddit_submissions.csv'
	glove_vector_location = '../glove.6B/glove.6B.' + str(glove_vector_dimensions) + 'd.txt'
	glove_vectors_dict = build_glove_dict(glove_vector_location)
	X, Y = build_minibatch(minibatch_size, glove_vectors_dict, data_file_location)


if __name__ == "__main__":
	main()