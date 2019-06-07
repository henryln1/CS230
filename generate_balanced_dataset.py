
import csv
import random
random.seed(1)

full_data_file = 'train_data.csv'
full_dev_data_file = 'dev_data.csv'
full_test_data_file = 'test_data.csv'
balanced_data_file = 'train_data_data_augmentation_and_max_score_capped=1000_60k_examples.csv'


dev_file = 'dev_data_8k_max_score_capped=1000.csv'
test_file = 'test_file_8k_max_score_capped=1000.csv'
max_number = 8000

threshold = 1000 #max score

title = None
dev_list = []
test_list = []

with open(full_dev_data_file, newline = '') as f:
	reader = csv.reader(f, delimiter = ',')
	counter = 0
	for line in reader:
		# print(line)
		if counter == 0:
			title = line
			counter += 1
			continue
		counter += 1
		if int(line[3]) > threshold:
			line[3] = str(threshold)
		dev_list.append(line)

random.shuffle(dev_list)
dev_list = dev_list[:max_number]

with open(dev_file, 'w') as file:
	writer = csv.writer(file, delimiter = ',', quotechar = '"')
	writer.writerow(title)
	for i in range(len(dev_list)):
		writer.writerow(dev_list[i])





with open(full_test_data_file, newline = '') as f:
	reader = csv.reader(f, delimiter = ',')
	counter = 0
	for line in reader:
		# print(line)
		if counter == 0:
			title = line
			counter += 1
			continue
		counter += 1
		if int(line[3]) > threshold:
			line[3] = str(threshold)
		test_list.append(line)

random.shuffle(test_list)
test_list = test_list[:max_number]

with open(test_file, 'w') as file:
	writer = csv.writer(file, delimiter = ',', quotechar = '"')
	writer.writerow(title)
	for i in range(len(test_list)):
		writer.writerow(test_list[i])






title = None
small_scores_list = []
large_scores_list = []


with open(full_data_file, newline = '') as f:
	reader = csv.reader(f, delimiter = ',')
	counter = 0
	for line in reader:
		# print(line)
		if counter == 0:
			title = line
			counter += 1
			continue
		counter += 1
		if int(line[3]) > threshold:
			line[3] = str(threshold)
			large_scores_list.append(line)
		else:
			small_scores_list.append(line)

print("Shuffling")
random.shuffle(small_scores_list)

max_number_small_score_examples = 50000
small_scores_list = small_scores_list[:max_number_small_score_examples]

while len(large_scores_list) < 10000:
	large_scores_list += large_scores_list

print("Length of Large: ", len(large_scores_list))
print("Length of Small: ", len(small_scores_list))

complete_list = small_scores_list + large_scores_list
random.shuffle(complete_list)

with open(balanced_data_file, 'w') as file:
	writer = csv.writer(file, delimiter = ',', quotechar = '"')
	writer.writerow(title)
	for i in range(len(complete_list)):
		writer.writerow(complete_list[i])


