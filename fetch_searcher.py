import numpy as np
import pandas as pd
import fasttext
import warnings
fasttext.FastText.eprint = lambda x: None


# Read data from CSV files into dataframes
df1 = pd.read_csv("data/brand_category.csv")
df2 = pd.read_csv("data/categories.csv")
df3 = pd.read_csv("data/offer_retailer.csv")

# Data preprocessing for df3 dataframe
df3['OFFER'] = df3['OFFER'].str.strip()  # Remove leading and trailing whitespaces
df3['OFFER'] = df3['OFFER'].str.lower()  # Convert 'OFFER' column values to lowercase
df3['RETAILER'] = df3['RETAILER'].fillna('')  # Fill missing values in 'RETAILER' column with empty string
df3['BRAND'] = df3['BRAND'].fillna('')  # Fill missing values in 'BRAND' column with empty string

# Group by 'OFFER' column and aggregate 'BRAND' and 'RETAILER' values
df3 = df3.groupby('OFFER').agg({
    'BRAND': lambda x: ' '.join(pd.Series(x).drop_duplicates()),  # Join unique 'BRAND' values with space
    'RETAILER': lambda x: ' '.join(pd.Series(x).drop_duplicates())  # Join unique 'RETAILER' values with space
}).reset_index()  # Reset index after grouping

# Create dictionaries mapping 'OFFER' to 'BRAND' and 'OFFER' to 'RETAILER'
offer_retailer_dict = dict(zip(df3['OFFER'], df3['RETAILER']))
offer_brand_dict = dict(zip(df3['OFFER'], df3['BRAND']))

# Merge df1 and df2 dataframes based on columns 'BRAND_BELONGS_TO_CATEGORY' and 'PRODUCT_CATEGORY'
merged_df = pd.merge(df1, df2, left_on='BRAND_BELONGS_TO_CATEGORY', right_on='PRODUCT_CATEGORY', how='left')

# Group by 'BRAND' column and aggregate 'PRODUCT_CATEGORY' and 'IS_CHILD_CATEGORY_TO' values
grouped_df = merged_df.groupby('BRAND').agg({
    'PRODUCT_CATEGORY': lambda x: ' '.join(pd.Series(x).drop_duplicates()),  # Join unique 'PRODUCT_CATEGORY' values with space
    'IS_CHILD_CATEGORY_TO': lambda x: ' '.join(pd.Series(x).drop_duplicates())  # Join unique 'IS_CHILD_CATEGORY_TO' values with space
}).reset_index()  # Reset index after grouping

# Merge df3 and grouped_df dataframes based on 'BRAND' column
merged_df = pd.merge(df3, grouped_df, on='BRAND', how='left')

# Fill missing values in specific columns
merged_df['RETAILER'] = merged_df['RETAILER'].fillna('')  # Fill missing values in 'RETAILER' column with empty string
merged_df['PRODUCT_CATEGORY'] = merged_df['PRODUCT_CATEGORY'].fillna('')  # Fill missing values in 'PRODUCT_CATEGORY' column with empty string
merged_df['IS_CHILD_CATEGORY_TO'] = merged_df['IS_CHILD_CATEGORY_TO'].fillna('')  # Fill missing values in 'IS_CHILD_CATEGORY_TO' column with empty string
merged_df['BRAND'] = merged_df['BRAND'].fillna('')  # Fill missing values in 'BRAND' column with empty string

# Group by 'OFFER' column and aggregate 'BRAND', 'PRODUCT_CATEGORY', 'IS_CHILD_CATEGORY_TO', and 'RETAILER' values
grouped_df = merged_df.groupby('OFFER').agg({
    'BRAND': lambda x: ', '.join(pd.Series(x).drop_duplicates()),  # Join unique 'BRAND' values with comma
    'PRODUCT_CATEGORY': lambda x: ', '.join(pd.Series(x).drop_duplicates()),  # Join unique 'PRODUCT_CATEGORY' values with comma
    'IS_CHILD_CATEGORY_TO': lambda x: ', '.join(pd.Series(x).drop_duplicates()),  # Join unique 'IS_CHILD_CATEGORY_TO' values with comma
    'RETAILER': lambda x: ', '.join(pd.Series(x).drop_duplicates())  # Join unique 'RETAILER' values with comma
}).reset_index()  # Reset index after grouping

# Create 'OFFER_TEXT' column by combining various columns with space and comma separators
grouped_df['OFFER_TEXT'] = grouped_df.apply(lambda x: '%s %s %s %s %s' % (x['OFFER'], x['BRAND'], x['IS_CHILD_CATEGORY_TO'], x['BRAND'], x['RETAILER']), axis=1)

# Create dictionary mapping 'OFFER' to 'OFFER_TEXT'
offer_dict = dict(zip(grouped_df['OFFER'], grouped_df['OFFER_TEXT']))


# Load fastText model
fasttext_model_path = './cc.en.300.bin'
fasttext_model = fasttext.load_model(fasttext_model_path)
def get_phrase_vector_fasttext(phrase, fasttext_model):
    """
    Get the vector representation of a phrase using fastText model.

    Parameters:
    phrase (str): The input phrase to be vectorized.
    fasttext_model: The pre-trained fastText model.

    Returns:
    numpy.ndarray: Vector representation of the input phrase.
    """
    # Get the vector for the entire phrase
    phrase_vector = fasttext_model.get_sentence_vector(phrase)
    return phrase_vector

def get_matches(phrase1, phrase2):
    """
    Calculate the dot product of vectors representing two phrases.

    Parameters:
    phrase1 (str): First input phrase.
    phrase2 (str): Second input phrase.

    Returns:
    float: Dot product of the vectors representing the input phrases.
    """
    vect1 = get_phrase_vector_fasttext(phrase1, fasttext_model)
    vect2 = get_phrase_vector_fasttext(phrase2, fasttext_model)
    return np.dot(vect1, vect2)
    #return cosine_similarity(phrase1, phrase2)

def get_best_offers(search_term):
    """
    Get the best matching offers based on the input search term.

    Parameters:
    search_term (str): The search term entered by the user.

    Returns:
    list: List of tuples containing the best matching offers and their scores, sorted in descending order of scores.
    """
    final_result = []
    for i in offer_dict.keys():
        offer_text = offer_dict[i]
        offer_text = offer_text.replace('\n', '')
        offer_retailer_text = offer_retailer_dict[i]
        offer_brand_text = offer_brand_dict[i]
        score1 = get_matches(search_term, offer_text)
        score2 = get_matches(search_term, offer_retailer_text)
        i = i.replace('\n', '')
        score3 = get_matches(search_term, i)
        ans = 0.4 * score1 + 0.4 * score2 + 0.2 * score3
        if ans > 0.10 and score3 > 0.10:
            final_result.append((i, ans))
    final_result = sorted(final_result, key=lambda x: x[1], reverse=True)
    return final_result

# Get user input for search term
query = input("Enter your search term: ")

# Get and display the best matching offers
ans = get_best_offers(query)


for i in range(20):
    print(ans[i])

