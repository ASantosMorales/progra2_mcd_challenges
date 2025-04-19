from preprocessing import dataset

def main():
	df = dataset()
	data = df.import_data(rows = 10, seed = 42)
	columns = ['headline', 'pros']
	data_cleaned = df.text_cleaning(df = data,
									columns = columns,
									lower_case = True,
									remove_numbers = True,
									remove_punctuation = True,
									remove_extra_spaces = True,
									remove_stopwords = True)
	print(data_cleaned)

if __name__ == '__main__':
	main()