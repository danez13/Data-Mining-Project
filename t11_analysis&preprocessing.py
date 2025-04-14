import pandas as pd
def analyze_data(data:pd.DataFrame):
    data_describe_numerical = {"features": [], "data type":[], "mean": [], "std": [], "min": [], "max": [], "range":[], "skewness":[], "kurtosis":[], "missing %": []}
    data_describe_categorical = {"features:": [], "data type":[], "unique values": [], "most frequent value": [], "most frequent value count": [], "missing %": []}
    for index, (feature, typing) in enumerate(data.dtypes.items()):
        if index < 6:
            data_describe_categorical["features:"].append(feature)
            data_describe_categorical["data type"].append(typing)
            data_describe_categorical["unique values"].append(data[feature].nunique())
            data_describe_categorical["most frequent value"].append(data[feature].mode()[0])
            data_describe_categorical["most frequent value count"].append(data[feature].value_counts().max())
            data_describe_categorical["missing %"].append(data[feature].isna().mean() * 100)
        else:
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
    descritive_numerical_data = pd.DataFrame(data_describe_numerical)
    descritive_categorical_data = pd.DataFrame(data_describe_categorical)
    descritive_categorical_data.to_csv("categorical_data_description.csv", index=False)
    descritive_numerical_data.to_csv("numerical_data_description.csv", index=False)
    return descritive_numerical_data, descritive_categorical_data

def preprocess_data(data:pd.DataFrame):
    data['label'] = data['f_FPro_class'].apply(lambda x: 0 if x == 3 else 1)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data.drop(columns=["brand", "food category", "store", "name", "original_ID"], inplace=True)
    return data

    #data['f_FPro_class'] = data['f_FPro_class'].apply(lambda x: 0 if x == 3 else 1)
    #data.dropna(inplace=True)
    #data.drop_duplicates(inplace=True)
    #data.drop(columns=["brand", "food category", "store", "name", "original_ID"], inplace=True)
    #return data

if __name__ == "__main__":
    data = pd.read_csv("product_data.csv")

    numerical_description,categorical_description = analyze_data(data)
    print("Numerical data description:")
    print(numerical_description)
    print("Categorical data description:")
    print(categorical_description)
    
    data = preprocess_data(data)
    data.to_csv("product_data_cleaned.csv", index=False)
