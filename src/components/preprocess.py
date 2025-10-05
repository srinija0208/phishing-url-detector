import pandas as pd


def combine_prepare_data(phishing_url_path, general_url_path):

    try:

        # Load datasets
        phishing_df = pd.read_csv(phishing_url_path)
        general_df = pd.read_csv(general_url_path)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    
    # rename columns to ensure consistency

    phishing_df.rename(columns={'url': 'URL','Type':'label'}, inplace=True)
    general_df.rename(columns={'url': 'URL','type':'label'}, inplace=True)

    # combine datasets
    combined_df = pd.concat([phishing_df, general_df], ignore_index=True)
    combined_df['label'] = combined_df['label'].str.lower()

    # Map labels to binary values
    combined_df['label'] = combined_df['label'].map({'legitimate':0, 'phishing':1})

    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)

    # Handle missing values by dropping rows with any missing values
    combined_df.dropna(inplace=True)

    return combined_df