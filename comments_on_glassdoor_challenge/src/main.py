from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocessing import *
from model_training import *
from evaluation import model_predictions, model_evaluation

def main():
	df = load_dataset()

	# Select the text columns to be processed
	columns = ['headline', 'pros', 'cons']

	df_cleaned, name_columns_cleaned = cleanning(df = df,
												columns = columns,
												lower_case = True,
												remove_numbers = True,
												remove_punctuation = True,
												remove_extra_spaces = True)
	
	df_no_stopwords, name_columns_no_stopwords = stopwords_remotion(df = df_cleaned,
																	columns = name_columns_cleaned) 
	
	df_lemmatized, name_columns_lemmatized = lemmatizing(df = df_no_stopwords,
														columns = name_columns_no_stopwords)

	df_text_merged, name_merged_column = text_columns_merging(df = df_lemmatized,
															columns=name_columns_lemmatized)

	columns_to_map = ['recommend']
	df_lemmatized_mapped, name_columns_mapped = rank_mapping(df = df_text_merged,
															columns = columns_to_map)

	X = tfidf_vectorizing(df_lemmatized_mapped[name_merged_column])
	y = df_lemmatized_mapped[name_columns_mapped[0]]
	X_train, X_test, y_train, y_test = get_train_test_subsets(X=X, y=y)

	model = model_training(model = LogisticRegression(max_iter=1000),
							X_train = X_train,
							y_train = y_train)

	y_pred, y_prob = model_predictions(model = model, 
										X_test = X_test,
										prob = True)

	model_evaluation(y_pred, y_prob, y_test, auc=False)

if __name__ == '__main__':
	main()