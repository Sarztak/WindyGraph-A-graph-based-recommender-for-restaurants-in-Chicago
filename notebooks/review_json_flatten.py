import json
import pandas as pd

def flatten_restaurant_reviews(json_data):
    """
    Flatten restaurant reviews from a nested JSON structure into a pandas DataFrame.
    Each row will contain a review with its associated restaurant ID.
    """
    flattened_data = []
    
    # Iterate through each restaurant and its reviews
    for restaurant_id, reviews in json_data.items():
        for review in reviews:
            # Create a flat record with restaurant_id and all review data
            flat_record = {
                'restaurant_id': restaurant_id,
                'review_id': review['id'],
                'rating': review['rating'],
                'text': review['text'],
                'time_created': review['time_created'],
                'url': review['url'],
                'user_id': review['user']['id'],
                'user_name': review['user']['name'],
                'user_profile_url': review['user']['profile_url'],
                'user_image_url': review['user']['image_url']
            }
            flattened_data.append(flat_record)
    
    # Convert to DataFrame
    df = pd.DataFrame(flattened_data)
    return df


if __name__ == "__main__":
    with open(r'data\reviews.json', 'r') as f:
        data = json.load(f)

    # 2. Convert to DataFrame
    reviews_df = flatten_restaurant_reviews(data)

    # save the file
    reviews_df.to_pickle('review.pickle')
    print(reviews_df.head())