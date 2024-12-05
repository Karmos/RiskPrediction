from src.eda import load_data, basic_eda, plot_missing_data, plot_feature_distribution
from src.preprocessing import preprocess_data, split_data, scale_data
from src.model import train_model, evaluate_model

def main():
    # Step 1: Load data
    df = load_data()

    columns_to_drop = [
        'id', 'member_id', 'url', 'issue_d', 'earliest_cr_line', 'last_credit_pull_d',
        'next_pymnt_d', 'emp_title', 'title', 'zip_code', 'addr_state', 'last_pymnt_d'
    ]
    df.drop(columns=columns_to_drop, inplace=True)

    plot_missing_data(df)

    # Remove rows with application type joint
    df = df[df['application_type'] != 'JOINT']
    # Drop columns related to 'joint' and the 'application_type' column
    columns_to_drop = [col for col in df.columns if 'joint' in col] + ['application_type']

    df.drop(columns=columns_to_drop, inplace=True)

    # Use list comprehension to identify columns with more than 30% missing values
    columns_to_drop = [col for col in df.columns if df[col].isna().mean() > 0.4]

    df.drop(columns=columns_to_drop, inplace=True)

    # Remove rows with pymnt_plan = 'y'
    df = df[df['pymnt_plan'] != 'y']
    df.drop(columns=['pymnt_plan', 'policy_code'], inplace=True)

    undSampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_train, y_train = undSampler.fit_resample(X_train, y_train)


















    plot_feature_distribution(data, 'some_feature')  # Replace with actual feature name

    # Step 3: Preprocess data
    processed_data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(processed_data, 'target_column')
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Step 4: Train the model
    model = train_model(X_train_scaled, y_train)

    # Step 5: Evaluate the model
    accuracy, cm = evaluate_model(model, X_test_scaled, y_test)


if __name__ == "__main__":
    main()