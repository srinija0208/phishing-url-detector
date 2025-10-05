import pandas as pd
import string
import re


def extract_features(df):

    df = df.copy()

    # Feature 1: Length of URL
    df['url_length'] = df['URL'].apply(len)

    # Feature 2: Count of special characters
    df['special_char_count'] = df['URL'].apply(lambda x: sum(i in string.punctuation for i in x))

    # Feature 3: Count of digits in URL
    df['digit_count'] = df['URL'].apply(lambda x: sum(i.isdigit() for i in x))

    # Feature 4: subdomain count in URL
    df['subdomain_count'] = df['URL'].apply(lambda x: x.count('.'))

    # Feature 5: is domain an IP address
    df["is_ip"] = df["URL"].apply(lambda x: 1 if re.match(r"(\d{1,3}\.){3}\d{1,3}", x) else 0)

    # Feature 6: Presence of 'https' in URL
    df['has_https'] = df['URL'].apply(lambda x: 1 if 'https' in x else 0)

    return df

# data = extract_features(data)