import sys
import numpy as np
import random
import re
import csv
import pickle

EXAMPLE_LENGTH = 100

def save_large_pickle(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def load_large_pickle(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj

# Split data into train, dev, test
def get_data(src, split=[0.8, 0.1, 0.1]):
	pass

def build_minibatch(minibatch_size, glove_vectors, data_file_location):
	glove_vector_dim = len(glove_vectors['the'])
	X = np.zeros(shape = (minibatch_size, EXAMPLE_LENGTH, glove_vector_dim))
	Y = np.zeros(shape = (minibatch_size, 1))
	num_data_points = num_lines = sum(1 for line in open(data_file_location, encoding="utf8"))
	random_rows = [random.randint(1, num_data_points) for x in range(minibatch_size)]
	data_points = []
	with open(data_file_location, encoding="utf8") as fd:
		reader = csv.reader(fd)
		data_points = [row for idx, row in enumerate(reader) if idx in random_rows]
	for i in range(len(data_points)):
		_, subreddit, title, score, num_comments, timestamp = tuple(data_points[i])
		title_words_list = re.sub(r'[^a-zA-Z ]', '', title).split()
		title_words_list = [x.lower() for x in title_words_list]
		for j in range(len(title_words_list)):
			curr_word = title_words_list[j]
			if curr_word in glove_vectors:
				X[i][j] = glove_vectors[curr_word]
		Y[i] = score
	return X, Y




def build_glove_dict(file_name):
	glove_dict = {}
	with open(file_name, 'r', encoding="utf8") as f:
		for line in f:
			line_list = line.split(" ")
			word = line_list[0]
			embeddings = line_list[1:]
			assert(str(len(embeddings)) in file_name) #making sure dimensions match file name
			embeddings = [float(x) for x in embeddings]
			glove_dict[word] = embeddings
	f.close()
	return glove_dict

<<<<<<< HEAD
def export_main(glove_dims, mb_size, split):
=======
def export_glove(glove_dims):
	glove_vector_dimensions = glove_dims
	# minibatch_size = mb_size
	data_file_location = './preprocessing_code/RS_2018-09_AskReddit_submissions.csv'
	glove_vector_location = './preprocessing_code/glove.6B/glove.6B.' + str(glove_vector_dimensions) + 'd.txt'
	glove_vectors_dict = build_glove_dict(glove_vector_location)
	# X, Y = build_minibatch(minibatch_size, glove_vectors_dict, data_file_location)
	# return X, Y
	return glove_vectors_dict	

def export_main(glove_dims, mb_size):
>>>>>>> 813298599663811342f43782bdd11c988e8db668
	# assert len(sys.argv) == 3 #make sure we are given a glove vector dim and minibatch size
	glove_vector_dimensions = glove_dims
	minibatch_size = mb_size
	if split == 'train':
		data_file_location = 'train_data.csv'
	elif split == 'dev':
		data_file_location = 'dev_data.csv'
	elif split == 'test':
		data_file_location = 'test_data.csv'
	glove_vector_location = './preprocessing_code/glove.6B/glove.6B.' + str(glove_vector_dimensions) + 'd.txt'
	glove_vectors_dict = build_glove_dict(glove_vector_location)
	X, Y = build_minibatch(minibatch_size, glove_vectors_dict, data_file_location)
	return X, Y

def main():
	assert len(sys.argv) == 3 #make sure we are given a glove vector dim and minibatch size
	glove_vector_dimensions = sys.argv[1]
	minibatch_size = int(sys.argv[2])
	data_file_location = './preprocessing_code/RS_2018-09_AskReddit_submissions_commas_removed.csv'
	glove_vector_location = './preprocessing_code/glove.6B/glove.6B.' + str(glove_vector_dimensions) + 'd.txt'
	glove_vectors_dict = build_glove_dict(glove_vector_location)
	X, Y = build_minibatch(minibatch_size, glove_vectors_dict, data_file_location)


if __name__ == "__main__":
	main()