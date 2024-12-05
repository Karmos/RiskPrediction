import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import IterativeImputer

def imputing(df):
    """Function to preprocess the data, handle missing values, etc."""
    if df.isnull().any().any():
        # Initialize the IterativeImputer with optimized settings
        imputer = IterativeImputer(max_iter=10, random_state=42, n_nearest_features=5)
        # Apply the IterativeImputer to the DataFrame
        imputed_data = imputer.fit_transform(df)
        # Convert the imputed data back to a DataFrame
        imputed_df = pd.DataFrame(imputed_data, columns=df.columns)
        imputed_df.to_csv('./dataset/preprocessed_data.csv', index=False)
    return imputed_df


def encode_data(df, ):
    """Function to split data into features and target."""
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    df['term'] = label_encoder.fit_transform(df['term'])
    df['initial_list_status'] = label_encoder.fit_transform(df['initial_list_status'])
    # %%
    df['verification_status'] = df['verification_status'].apply(verification_status)
    # %%
    categorical_features = df.select_dtypes(include='object').drop(columns=['loan_status'])
    encoded_features = pd.get_dummies(categorical_features, dtype=int)
    df = pd.concat([df, encoded_features], axis=1)
    # %%
    df.drop(columns=categorical_features.columns, inplace=True)
    return df


def sub_grades_encoding(df):
    # Define a dictionary mapping letter grades to numeric values
    grade_map = {
        'A': 7,
        'B': 6,
        'C': 5,
        'D': 4,
        'E': 3,
        'F': 2,
        'G': 1
    }

    # Vectorized operation for letter grade encoding
    grade_vals = df['sub_grade'].str.extract('([A-G])')[0].map(grade_map)

    # Vectorized operation for numeric adjustment based on 1-5
    numeric_adjustments = {
        '1': 0.8,
        '2': 0.6,
        '3': 0.4,
        '4': 0.2,
        '5': 0.0
    }

    # Extract the numeric suffix (1, 2, 3, etc.) and apply the corresponding adjustment
    numeric_vals = df['sub_grade'].str.extract('([1-5])')[0].map(numeric_adjustments).fillna(0)

    # Combine the grade and numeric adjustments
    df['sub_grade'] = grade_vals + numeric_vals

    return df

# verification status encoding
def verification_status(x):
    # Return 0 for 'Not Verified' and 1 for any other status
    return 0 if x == 'Not Verified' else 1


def scale_data(X_train, X_test, X):
    """Function to scale the features."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    return X_train, X_test