import pandas as pd
import numpy as np

def preprocess_data(data_source):
    """
    Preprocess the influencer dataset.
    
    Parameters:
    data_source: Either a path to a CSV file or a pandas DataFrame
    
    Returns:
    pd.DataFrame: Processed dataframe ready for the ranking algorithm
    """
    try:
        # Check if data_source is a file path or DataFrame
        if isinstance(data_source, str):
            # Load the dataset from file path
            df = pd.read_csv(data_source)
        else:
            # Assume it's already a DataFrame
            df = data_source.copy()
        
        # Convert metrics to numeric values
        # Handle 'k', 'm', and 'b' suffixes
        def convert_to_numeric(value):
            if isinstance(value, str):
                if 'k' in value.lower():
                    return float(value.lower().replace('k', '')) * 1000
                elif 'm' in value.lower():
                    return float(value.lower().replace('m', '')) * 1000000
                elif 'b' in value.lower():
                    return float(value.lower().replace('b', '')) * 1000000000
            return value
        
        # Apply conversion to relevant columns
        for col in ['posts', 'followers', 'avg_likes', 'new_post_avg_like', 'total_likes']:
            if col in df.columns:
                df[col] = df[col].apply(convert_to_numeric)
        
        # Convert engagement rate from percentage to decimal
        if '60_day_eng_rate' in df.columns:
            if df['60_day_eng_rate'].dtype == object:  # If it's a string
                df['60_day_eng_rate'] = df['60_day_eng_rate'].str.rstrip('%').astype('float') / 100
        
        # Fill missing country values
        if 'country' in df.columns:
            df['country'] = df['country'].fillna('Unknown')
        
        # Calculate additional metrics for our algorithm
        
        # Normalize the existing influence score to range 0-1
        if 'influence_score' in df.columns:
            df['norm_influence_score'] = df['influence_score'] / 100
        
        # Create credibility score
        # For now, we'll use a combination of engagement rate and followers
        if '60_day_eng_rate' in df.columns and 'followers' in df.columns:
            df['credibility_score'] = (df['60_day_eng_rate'] * 5 + 
                                      np.log10(df['followers'] + 1) / 9)
            # Cap at 1.0
            df['credibility_score'] = df['credibility_score'].clip(0, 1)
        else:
            df['credibility_score'] = 0.5  # Default value
        
        # Create longevity score based on total posts
        if 'posts' in df.columns:
            df['longevity_score'] = np.log10(df['posts'] + 1) / 4
            df['longevity_score'] = df['longevity_score'].clip(0, 1)
        else:
            df['longevity_score'] = 0.5  # Default value
        
        # Create engagement quality score
        if all(col in df.columns for col in ['60_day_eng_rate', 'avg_likes', 'followers']):
            # Avoid division by zero by adding 1
            df['engagement_score'] = (df['60_day_eng_rate'] * 0.7 + 
                                    np.log10(df['avg_likes'] + 1) / np.log10(df['followers'] + 10) * 0.3)
            df['engagement_score'] = df['engagement_score'].clip(0, 1)
        else:
            df['engagement_score'] = 0.5  # Default value
        
        # Add a column for fake fame detection flag
        df['fake_fame_flag'] = 0
        
        print("Preprocessing completed successfully!")
        return df
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise