import pandas as pd

def main():
    reviews_df = pd.read_pickle('data/processed_review_data.pkl')
    restaurants_df = pd.read_pickle('data/processed_restaurant_data.pkl')
    reviews_df.to_parquet('data/processed_review_data.parquet')
    restaurants_df.to_parquet('data/processed_restaurant_data.parquet')
    reviews_df.to_csv('data/processed_review_data.csv')
    restaurants_df.to_csv('data/processed_restaurant_data.csv')

    print(reviews_df.columns)
    print(restaurants_df.columns)


if __name__ == "__main__":
    main()

   

