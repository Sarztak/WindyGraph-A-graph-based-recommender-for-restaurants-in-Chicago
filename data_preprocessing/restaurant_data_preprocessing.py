import pandas as pd
import numpy as np
import json
import pickle
import os
import math
from ast import literal_eval

# Function to safely convert string representation of categories to list
def parse_categories(categories_str):
    try:
        if isinstance(categories_str, str):
            # Parse the string representation to Python object
            cats = literal_eval(categories_str)
            # Extract just the category titles
            return [cat['title'] for cat in cats]
        elif isinstance(categories_str, list):
            # Already in list format
            return [cat['title'] for cat in cats]
        else:
            return []
    except:
        return []
# Using a simple projection centered on Chicago


def local_projection(lat, lon):
    chicago_center_lat, chicago_center_lon = 41.8781, -87.6298
    # Constants for converting degrees to approximate kilometers at Chicago's latitude
    lat_km = 111.0  # 1 degree latitude ≈ 111 km
    lon_km = 111.0 * math.cos(math.radians(chicago_center_lat))  # Longitude scale factor
    
    # Calculate distance from center in kilometers
    x = (lon - chicago_center_lon) * lon_km
    y = (lat - chicago_center_lat) * lat_km
    
    # Normalize to [-1, 1] based on city boundaries (adjust constants as needed)
    max_dist = 40  # ~40km covers most of Chicago
    x_norm = (x / max_dist) * 0.5
    y_norm = (y / max_dist) * 0.5
    
    return x_norm, y_norm

def process_restaurant_data(input_file, output_file):
    """
    Process restaurant data for GNN model
    
    Args:
        input_file: Path to the input CSV or JSON file
        output_file: Path to save the processed pickle file
    """
    print(f"Loading data from {input_file}...")
    
    # Load data based on file extension
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    elif input_file.endswith('.json'):
        df = pd.read_json(input_file)
    else:
        raise ValueError("Input file must be CSV or JSON")
    
    print(f"Original data shape: {df.shape}")
    
    # Select relevant columns
    relevant_cols = [
        'id',                    # Unique identifier
        'name',                  # Restaurant name
        'categories',            # For category-restaurant edges
        'rating',                # Node feature
        'review_count',          # Node feature
        'coordinates.latitude',  # For location filtering
        'coordinates.longitude', # For location filtering
    ]
    
    # Keep only columns that exist in the dataframe
    cols_to_keep = [col for col in relevant_cols if col in df.columns]
    df_clean = df[cols_to_keep].copy()
    
    print(f"Selected {len(cols_to_keep)} columns")
    
    # Handle missing values
    df_clean.dropna(subset=['id', 'name', 'categories'], inplace=True)
    print(f"Data shape after dropping rows with missing essential info: {df_clean.shape}")
    
    # Fill missing values for non-essential columns
    # Do not recommend filling with zeros; fortunately there are not missing values
    # if 'rating' in df_clean.columns:
    #     df_clean['rating'].fillna(0, inplace=True)
    # if 'review_count' in df_clean.columns:
    #     df_clean['review_count'].fillna(0, inplace=True)
    
    # Process categories
    df_clean['categories_list'] = df_clean['categories'].apply(parse_categories)
    
    # Log transform review count (handle zeros)
    df_clean['log_review_count'] = np.log1p(df_clean['review_count'])
    
    # Normalize ratings to [0,1] range
    min_rating = df_clean['rating'].min()
    max_rating = df_clean['rating'].max()

    # Min-max normalization
    df_clean['normalized_rating'] = (df_clean['rating'] - min_rating) / max_rating  
    
    # Alternative: Z-score normalization
    # df_clean['normalized_rating'] = (df_clean['rating'] - df_clean['rating'].mean()) / df_clean['rating'].std()
    
    # Create popularity feature combining rating and review count

    # Normalize log_review_count to [0,1]
    max_log_reviews = df_clean['log_review_count'].max()

    df_clean['normalized_log_review_count'] = df_clean['log_review_count'] / max_log_reviews


    '''

    every feature in your model implicitly encodes assumptions about what matters to users. Being intentional about these choices - rather than just implementing what seems mathematically sound - leads to recommendations that feel more human and intuitive.
    
    Impact on Recommendations
    what each version of popularity score would emphasize in recommendations:
    Version 1 (Multiplication):
    Would heavily favor established "institution" restaurants
    New but promising places would be buried
    Creates a "winner-takes-all" effect where a few restaurants dominate
    Emphasizes consensus favorites

    Version 2 (Weighted Average):
    Gives more balanced recommendations
    Allows newer places with good ratings to emerge faster
    Creates more diverse recommendations
    Better represents different types of popularity

    The 0.7 weight on review count and 0.3 on rating is a sensible default because more reviews means more confidence in the rating

    Ultimately, the best weights should be determined by how well they help your system make recommendations that users actually enjoy and act on. This might require some iteration once  the system running and can collect feedback.

    '''

    df_clean['popularity_score'] = (
        0.7 * df_clean['normalized_log_review_count'] + 
        0.3 * df_clean['normalized_rating']
    )
    # df_clean['popularity_score'] = df_clean['normalized_rating'] * df['norm_log_review_count']


    # Method 2: Alternative - Wilson score (simplified version)
    # This considers both rating and number of reviews statistically
    # Higher ratings with more reviews get higher scores
    # Addresses the "new restaurant problem" by accounting for statistical confidence
    
    df_clean['wilson_score'] = (
        (df_clean['rating'] * df_clean['review_count'] + 3.0 * 2) / 
        (df_clean['review_count'] + 2 * 2)
    )

    # Normalize wilson score to [0,1] range
    min_rating = df_clean['wilson_score'].min()
    max_rating = df_clean['wilson_score'].max()

    # Min-max normalization
    df_clean['normalized_wilson_score'] = (df_clean['wilson_score'] - min_rating) / max_rating
    
    # Rename coordinate columns for clarity

    df_clean.rename(columns={
        'coordinates.latitude': 'latitude',
        'coordinates.longitude': 'longitude'
    }, inplace=True)

    df_clean['lat_lon_projection'] = df_clean.apply(lambda x: local_projection(x['latitude'], x['longitude']), axis=1)

    df_clean['normalized_latitude'] = df_clean.apply(lambda x: x['lat_lon_projection'][0], axis=1)

    df_clean['normalized_longitude'] = df_clean.apply(lambda x: x['lat_lon_projection'][1], axis=1)

    df_clean.drop(columns='lat_lon_projection', inplace=True)

    # Create a unique list of all categories
    all_categories = []
    for cats in df_clean['categories_list']:
        all_categories.extend(cats)
    unique_categories = sorted(list(set(all_categories)))
    
    print(f"Found {len(unique_categories)} unique categories")
    
    # Create mapping dictionaries for categories and restaurants
    category_to_id = {category: idx for idx, category in enumerate(unique_categories)}
    restaurant_to_id = {rest_id: idx for idx, rest_id in enumerate(df_clean['id'].unique())}
    
    # drop category column
    df_clean.drop(columns='categories', inplace=True)
    
    # Create the final dataframe with processed data
    result = {
        'restaurants': df_clean,
        'category_to_id': category_to_id,
        'restaurant_to_id': restaurant_to_id,
        'unique_categories': unique_categories
    }
    
    # Save the processed data
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Processed data saved to {output_file}")
    return result

if __name__ == "__main__":
    # Example usage
    input_file = "data/yelp_restaurant_data.csv"  # Replace with your input file
    output_file = "data/processed_restaurant_data.pkl"
    
    # Process the data
    result = process_restaurant_data(input_file, output_file)
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Number of restaurants: {len(result['restaurants'])}")
    print(f"Number of categories: {len(result['unique_categories'])}")
    
    # Display first few rows of processed data
    print("\nSample of processed data:")
    print(result['restaurants'][['id', 'name', 'categories_list', 'rating', 'normalized_latitude']].head())
