def main():
    print("Hello from windygraph-a-graph-based-recommender-for-restaurants-in-chicago!")


if __name__ == "__main__":
    # main()
    import pickle

    # Load the dictionary from the pickle file
    with open('data/processed_restaurant_data.pkl', 'rb') as file:
        loaded_results = pickle.load(file)

    # Accessing data from the loaded dictionary
    df_clean = loaded_results['restaurants']      # DataFrame
    category_to_id = loaded_results['category_to_id']  # Dictionary
    restaurant_to_id = loaded_results['restaurant_to_id']  # Dictionary
    unique_categories = loaded_results['unique_categories']  # List or dict

    print(df_clean.head())  # Check the DataFrame
    print(df_clean.columns)
    print(category_to_id)   # Check the dictionary

