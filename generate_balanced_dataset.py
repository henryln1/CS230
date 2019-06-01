
import csv
import random


full_data_file = 'train_data.csv'
balanced_data_file = 'train_data_data_augmentation_and_max_score_capped=1000.csv'
# balanced_data_file = 'train_data_data_augmentation.csv'

title = None
small_scores_list = []
large_scores_list = []
threshold = 1000


with open(full_data_file, newline = '') as f:
	reader = csv.reader(f, delimiter = ',')
	counter = 0
	for line in reader:
		print("Line ", counter)
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
random.seed(1)
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


