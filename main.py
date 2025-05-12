import pandas as pd

def main():
    reviews_df = pd.read_pickle('data/processed_review_data.pkl')
    restaurants_df = pd.read_pickle('data/processed_restaurant_data.pkl')

    print(reviews_df.columns)
    print(restaurants_df.columns)


if __name__ == "__main__":
    main()

   

