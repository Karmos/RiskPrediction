from src.utils import *
from src.eda import *
from src.preprocessing import *
from src.model import *
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def main():
    # Step 1: Load data
    dataset_path = './dataset/dataset.csv'
    df = load_data(dataset_path)

    columns_to_drop = [
        'id', 'member_id', 'url', 'issue_d', 'earliest_cr_line', 'last_credit_pull_d',
        'next_pymnt_d', 'emp_title', 'title', 'zip_code', 'addr_state', 'last_pymnt_d'
    ]
    df.drop(columns=columns_to_drop, inplace=True)

    #Step 2: EDA

    print_missing_data(df)

    plot_barplot(df, 'application_type')

    # Remove rows with application type joint
    df = df[df['application_type'] != 'JOINT']
    # Drop columns related to 'joint' and the 'application_type' column
    columns_to_drop = [col for col in df.columns if 'joint' in col] + ['application_type']
    df.drop(columns=columns_to_drop, inplace=True)

    plot_missing_data(df, 5)

    # Use list comprehension to identify columns with more than 30% missing values
    columns_to_drop = [col for col in df.columns if df[col].isna().mean() > 0.4]
    df.drop(columns=columns_to_drop, inplace=True)

    plot_missing_data(df, 5)

    print_uniques(df)

    # Remove rows with pymnt_plan = 'y'
    df = df[df['pymnt_plan'] != 'y']
    df.drop(columns=['pymnt_plan', 'policy_code', 'grade'], inplace=True)

    df = plot_emp_length_vs_loan_status(df)

    df.drop(columns=['emp_length'], inplace=True)

    plot_feature_distributions(df)

    plot_boxplot_interestrate(df)

    plot_boxplot_dti(df)

    plot_loan_status_vs_median_income(df)

    plot_loans_per_term_and_status(df)

    plot_loans_by_subgrade(df)

    loan_status_by_purpose(df)

    #coping with skewness
    df = transform_skewed_features(df)

    plot_feature_distributions(df)

    #encoding
    df = sub_grades_encoding(df)

    df = encode_data(df)

    df = process_loan_status(df)

    #impute data
    imputed_df = imputing(df)
    prep_dataset_path = './dataset/preprocessed_data.csv'
    imputed_df.to_csv(prep_dataset_path, index=False)


    # Step 3: Preprocess data
    df = imputed_df

    X = df.drop(columns=['risk_level'])
    y = df['risk_level']

    # Informazioni non disponibili per la credit risk
    X.columns
    columns_to_drop = [col for col in
                       ['out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
                        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt',
                        'total_rev_hi_lim']]

    X.drop(columns=columns_to_drop, inplace=True)

    # Eseguiamo il primo train-test split con stratificazione per ottenere i dati
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Scaling
    X_train, X_test = scale_data(X_train, X_test, X);

    # Bilanciamento
    undSampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_train, y_train = undSampler.fit_resample(X_train, y_train)

    #Correlazione e riduzione dimensionalità
    correlation_matrix = df.corr()
    print(correlation_matrix)

    X_train_copy = X_train
    X_test_copy = X_test
    X_train, X_test = pca_analysis(X_train, X_test, 0.9)

    # Step 4: Train the model

    # --- SVM ---
    svm = SVC(random_state=42)
    param_grid = {"C": [1], "kernel": ["linear", "rbf"]}
    best_svm_model, cv_results = train_with_grid_search(svm, param_grid, X_train, y_train)
    results = evaluate_model(best_svm_model, cv_results, X_test, y_test)
    save_grid_search_results(results,
                             ['param_C', 'param_kernel', 'mean_test_score', 'std_test_score', 'rank_test_score', 'precision', 'recall', 'f1_score', 'accuracy'],
                             'results/creditRisk/svm_results.csv')
    save_model(best_svm_model, 'models/creditRisk/best_svm_model.pkl')
    y_pred_svm = best_svm_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_svm, 'SVM_creditRisk')
    print("Classification Report (SVM):")
    print(classification_report(y_test, y_pred_svm))

    # --- Logistic Regression ---
    lr = LogisticRegression(random_state=42, max_iter=500)
    param_grid = {"penalty": ["l2"], "C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}
    best_lr_model, cv_results = train_with_grid_search(lr, param_grid, X_train, y_train)
    results = evaluate_model(best_lr_model, cv_results, X_test, y_test)
    save_grid_search_results(results,
                             ['param_penalty', 'param_C', 'param_solver', 'mean_test_score', 'std_test_score', 'rank_test_score', 'precision', 'recall', 'f1_score', 'accuracy'],
                             'results/creditRisk/lr_results.csv')
    save_model(best_lr_model, 'models/creditRisk/best_lr_model.pkl')
    y_pred_lr = best_lr_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_lr, 'logisticRegression_creditRisk')
    print("Classification Report (Logistic Regression):")
    print(classification_report(y_test, y_pred_lr))

    # --- Random Forest ---
    rf = RandomForestClassifier(random_state=42)
    param_grid = {"n_estimators": [100, 200], "max_depth": [5, 7], "min_samples_leaf": [1, 2]}
    best_rf_model, cv_results = train_with_grid_search(rf, param_grid, X_train, y_train)
    results = evaluate_model(best_rf_model, cv_results, X_test, y_test)
    save_grid_search_results(results, ['param_n_estimators', 'param_max_depth', 'param_min_samples_leaf', 'mean_test_score', 'std_test_score', 'rank_test_score', 'precision', 'recall', 'f1_score', 'accuracy'], 'results/creditRisk/rf_results.csv')
    save_model(best_rf_model, 'models/creditRisk/best_rf_model.pkl')
    y_pred_rf = best_rf_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_rf, 'randomForest_creditRisk')
    print("Classification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf))

    # --- XGBoost ---
    xgb = XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='logloss')
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
    }
    best_xgb_model, cv_results = train_with_grid_search(xgb, param_grid, X_train, y_train)
    results = evaluate_model(best_xgb_model, cv_results, X_test, y_test)
    save_grid_search_results(results,
                             ["param_n_estimators", "param_learning_rate", "param_max_depth", "param_subsample",
                              "mean_test_score", "std_test_score", "rank_test_score", 'precision', 'recall', 'f1_score', 'accuracy'], "results/creditRisk/xgb_results.csv")
    save_model(best_xgb_model, "models/creditRisk/best_xgb_model.pkl")
    y_pred_xgb = best_xgb_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_xgb, 'XGBoost_creditRisk')
    print("Classification Report (XGBoost):")
    print(classification_report(y_test, y_pred_xgb))

    # Step 4: SHAP
    analyze_rf_model(X_train_copy, X_test_copy, ['0', '1'], 'randomForest_creditRisk')

if __name__ == "__main__":
    main()