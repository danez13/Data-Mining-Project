import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Function to analyze and summarize numerical and categorical data
def analyze_data(data: pd.DataFrame):
    # Dictionaries to hold stats for numerical and categorical columns
    data_describe_numerical = {
        "features": [], "data type": [], "mean": [], "std": [],
        "min": [], "max": [], "range": [], "skewness": [],
        "kurtosis": [], "missing %": []
    }
    data_describe_categorical = {
        "features:": [], "data type": [], "unique values": [],
        "most frequent value": [], "most frequent value count": [],
        "missing %": []
    }

    # Loop through each column in the dataframe
    for index, (feature, typing) in enumerate(data.dtypes.items()):
        if index < 6:  # first 6 columns are categorical
            data_describe_categorical["features:"].append(feature)
            data_describe_categorical["data type"].append(typing)
            data_describe_categorical["unique values"].append(data[feature].nunique())
            data_describe_categorical["most frequent value"].append(data[feature].mode()[0])
            data_describe_categorical["most frequent value count"].append(data[feature].value_counts().max())
            data_describe_categorical["missing %"].append(data[feature].isna().mean() * 100)
        else:  # columns are considered numerical
            data_describe_numerical["features"].append(feature)
            data_describe_numerical["data type"].append(typing)
            data_describe_numerical["mean"].append(data[feature].mean())
            data_describe_numerical["std"].append(data[feature].std())
            data_describe_numerical["min"].append(data[feature].min())
            data_describe_numerical["max"].append(data[feature].max())
            data_describe_numerical["range"].append(data[feature].max() - data[feature].min())
            data_describe_numerical["skewness"].append(data[feature].skew())
            data_describe_numerical["kurtosis"].append(data[feature].kurtosis())
            data_describe_numerical["missing %"].append(data[feature].isna().mean() * 100)

    # Convert summaries to DataFrames
    descritive_numerical_data = pd.DataFrame(data_describe_numerical)
    descritive_categorical_data = pd.DataFrame(data_describe_categorical)

    # Export summaries to CSV files
    descritive_categorical_data.to_csv("t11_categorical_data_description.csv", index=False)
    descritive_numerical_data.to_csv("t11_numerical_data_description.csv", index=False)

    return descritive_numerical_data, descritive_categorical_data

# Function to preprocess data before modeling
def preprocess_data(data: pd.DataFrame):
    # Create a binary label from 'f_FPro_class'
    # Class 3 becomes 0, everything else becomes 1
    data['label'] = data['f_FPro_class'].apply(lambda x: 0 if x == 3 else 1)

    # Remove rows with missing values and duplicates
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)

    # Encode categorical columns using ordinal encoding
    encoder = OrdinalEncoder()
    data['store'] = encoder.fit_transform(data[['store']])
    data['food category'] = encoder.fit_transform(data[['food category']])

    # Drop columns not needed for modeling
    data.drop(columns=["f_FPro_class", "brand", "name", "original_ID"], inplace=True)

    return data

# Run the script if executed directly
if __name__ == "__main__":
    # Load raw dataset
    data = pd.read_csv("product_data.csv")

    # Analyze dataset and print summaries
    numerical_description, categorical_description = analyze_data(data)
    print("Numerical data description:")
    print(numerical_description)
    print("Categorical data description:")
    print(categorical_description)
    
    # Preprocess data and export cleaned version
    data = preprocess_data(data)
    data.to_csv("t11_cleaned_data.csv", index=False)
