from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocessing import *
from model_training import *
from evaluation import model_predictions, model_evaluation, words_frequency_plot
from utils import store_data, get_data, store_df

def main():
	"""
	df = load_dataset()

	# Select the text columns to remove empty rows
	columns = ['recommend', 'headline', 'pros', 'cons']
	df = remove_empty_rows(df, columns)

	df = remove_category_from_samples(df=df, column_name='recommend', category_label='o')

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
	df_merged_mapped, name_columns_mapped = rank_mapping(df = df_text_merged,
														columns = columns_to_map)

	X = tfidf_vectorizing(df_merged_mapped[name_merged_column])
	y = df_merged_mapped[name_columns_mapped[0]]
	X_train, X_test, y_train, y_test = get_train_test_subsets(X=X, y=y)

	model = model_training(model = LogisticRegression(max_iter=1000),
							X_train = X_train,
							y_train = y_train)

	store_data({'model':model, 'X_test':X_test, 'y_test':y_test}, 'data_processed')
	store_df(df_text_merged, 'df_text_merged')
	"""
	model, X_test, y_test = get_data('data_processed')
	df_text_merged = pd.read_csv(os.path.join(os.getcwd(), 'df_text_merged.csv'))
	print('Data retreived.')

	y_pred, y_prob = model_predictions(model = model, 
										X_test = X_test,
										prob = True)

	metrics_, artifacts_ = model_evaluation(y_pred, y_prob, y_test)
	artifacts_['words_frequency_graph'] = words_frequency_plot(df_text_merged['text_merged_column'])
	#"""
if __name__ == '__main__':
	main()