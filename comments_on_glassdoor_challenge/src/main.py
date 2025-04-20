from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocessing import dataset, text_preprocessing
from model_training import sentimentClassifier

def main():
	data = dataset()
	df = data.load()
	columns = ['headline', 'pros', 'cons']
	text = text_preprocessing()

	df_cleaned, name_columns_cleaned = text.cleanning(df = df,
													columns = columns,
													lower_case = True,
													remove_numbers = True,
													remove_punctuation = True,
													remove_extra_spaces = True)
	
	df_no_stopwords, name_columns_no_stopwords = text.stopwords_remotion(df = df_cleaned,
																		columns = name_columns_cleaned) 
	
	df_lemmatized, name_columns_lemmatized = text.lemmatizing(df = df_no_stopwords,
															columns = name_columns_no_stopwords)

	df_text_merged, name_merged_column = text.text_columns_merging(df = df_lemmatized,
																columns=name_columns_lemmatized)

	columns_to_map = ['recommend']
	df_lemmatized_mapped, name_columns_mapped = text.rank_mapping(df = df_text_merged,
																columns = columns_to_map)

	text_classifier = sentimentClassifier()
	X = text_classifier.tfidf_vectorizing(df_lemmatized_mapped[name_merged_column])
	y = df_lemmatized_mapped[name_columns_mapped[0]]
	X_train, X_test, y_train, y_test = text_classifier.split(X=X, y=y)

	model = text_classifier.model_fitting(model = LogisticRegression(max_iter=1000),
										X = X_train,
										y = y_train)
	y_pred = model.predict(X_test)

	accurracy = accuracy_score(y_test, y_pred)
	print(f'Accurracy score = {round(accurracy, 4)}')

if __name__ == '__main__':
	main()