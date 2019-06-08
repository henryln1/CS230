
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np 



def extract_results(file_name):
	predicted = []
	actual = []
	predicted_string = ""
	actual_string = ""
	with open(file_name, 'r') as f:
		line = f.readline()
		while line:
			print(line)
			if 'Predicted' in line:
				predicted_string = f.readline()
				predicted_string = predicted_string.replace(' ', '')
				predicted_string = predicted_string.replace('[', '')
				predicted_string = predicted_string.replace(']', '')
				predicted_string = predicted_string.replace('\n', '')
				predicted = predicted_string.split(',')
				if ' ' in predicted:
					predicted.remove(' ')
			if 'Actual' in line:
				actual_string = f.readline()
				actual_string = actual_string.replace(' ', '')
				actual_string = actual_string.replace('[', '')
				actual_string = actual_string.replace(']', '')
				actual_string = actual_string.replace('\n', '')
				actual = actual_string.split(',')
				if ' ' in actual:
					actual.remove(' ')
			line = f.readline()

	print("length predicted:", len(predicted))
	print('length actual: ', len(actual))
	print(set(predicted))
	assert len(predicted) == len(actual)
	print('Unique labels in predicted: ', set(predicted))
	print('Unique labels in actual: ', set(actual))
	return predicted, actual

def calculate_precision_recall(predicted, actual):
	matrix = confusion_matrix(actual, predicted)
	M_ii = matrix.diagonal()

	precision = M_ii / np.sum(matrix, axis = 0)
	recall = M_ii / np.sum(matrix, axis = 1)
	f1 = 2 * (precision * recall) / (precision + recall)
	return precision, recall, f1


def build_confusion_matrix(predicted, actual):
		# upper_bounds = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
	classes = ['0',
				'1',
				'2-5',
				'6-10',
				'11-20',
				'21-30',
				'31-40',
				'41-50',
				'51-60',
				'61-70',
				'71-100',
				'101-200',
				'201-300',
				'301-400',
				'401-500',
				'501-600',
				'601-700',
				'701-800',
				'801-900',
				'901-1000'
				,'1000+']

	cnf_matrix = confusion_matrix(actual, predicted)#, classes)
	cmap=plt.cm.Blues
	plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
	plt.title("Confusion Matrix for Viral Prediction")
	plt.gcf().subplots_adjust(bottom=0.18)
	#plt.colorbar()
	plt.colorbar().set_label("Number of Occurrences")
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.xlabel("Predicted Label")
	plt.ylabel("True Label")
	plt.yticks(tick_marks, classes)
	plt.savefig("confusion_matrix.png")
	plt.close()



def main():
	if len(sys.argv) > 1:
		file_name = sys.argv[1]
	else:
		file_name = 'classification_results_new_split.txt'

	predicted, actual = extract_results(file_name)
	print("Results extracted")
	build_confusion_matrix(predicted, actual)
	print("Confusion Matrix built")
	precision, recall, f1 = calculate_precision_recall(predicted, actual)
	print('Precision: ', precision)
	print('Recall: ', recall)
	print('F1: ', f1)




if __name__ == "__main__":
	main()



