from preprocessing import dataset, text_preprocessing

def main():
	data = dataset()
	df = data.load(rows = 10, seed = 42)
	columns = ['headline', 'pros']
	text = text_preprocessing()
	df_cleaned, name_columns_cleaned = text.cleanning(df = df,
													columns = columns,
													lower_case = True,
													remove_numbers = True,
													remove_punctuation = True,
													remove_extra_spaces = True)
	df_no_stopwords, name_columns_no_stopwords = text.stopwords_remotion(df = df_cleaned,
																		columns = name_columns_cleaned) 
	df_lemmatized, name_columns_lemmatized = text.lemmatizing(df = df_cleaned,
															columns = name_columns_no_stopwords)

if __name__ == '__main__':
	main()