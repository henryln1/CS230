
import sys




def extract_results(file_name):
	predicted = []
	actual = []
	with open(file_name, 'r') as f:
		for line in f:
			if 'Predicted' in line:
				pass
			elif 'Actual' in line:
				pass


def calculate_f1(predicted, actual):
	pass


def build_confusion_matrix(predicted, actual):
	pass


def main():
	if len(sys.argv) > 1:
		file_name = sys.argv[1]
	else:
		file_name = 'classification_results.txt'

	predicted, actual = extract_results(file_name)
	build_confusion_matrix(predicted, actual)
	calculate_f1(predicted, actual)





if __name__ == "__main__":
	main()



