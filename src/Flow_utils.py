"""

    This script collects utility functions to make the flow file smaller and more readable.

"""
import pandas as pd
import numpy as np

def replace_outliers_with_boundaries(X, num_features, multiplier=3):
    X_check = X[num_features].drop('sentiment_score_LM', axis = 1)
    Q1 = X_check.quantile(0.25)
    Q3 = X_check.quantile(0.75)
    IQR = Q3 - Q1

    # Identify outliers based on IQR with the specified multiplier
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    outliers_iqr = (X_check < lower_bound) | (X_check > upper_bound)
    
    # Filter DataFrame to include only outlier values
    outlier_values = X_check[outliers_iqr].stack()
    print("outlier values before replacement:")
    print(outlier_values)
    print(len(outlier_values))
    if len(outlier_values) == 0:
        print('There are no outliers')
    else:
        # Replace outliers with their respective boundaries
        for col in X_check.columns:
            X[col] = X[col].clip(lower=lower_bound[col], upper=upper_bound[col])

    return X

def feature_preprocess(data, target, num_features, cat_features):
    '''
    this function is to label categorical column and 
    generate shift and pecent change features based on origial features
    '''
    data2 = data[target + num_features+cat_features]
    data2 = data2.ffill() # use last value to fill nan
    
    # label sentiment and target
    sentiment_mapping = {
        'positive': 1,
        'negative': -1,
        'neutral': 0
    }
    target_mapping = {
        'increase': 1,
        'decrease': -1,
        'hold': 0
    }
    data = data.copy()
    if not all(item in [-1, 0, 1] for item in data2['Sentiment'].unique()):
        data2['Sentiment'] = data2['Sentiment'].map(sentiment_mapping)
    data2['Target_category'] = data2['Target_category'].map(target_mapping)

    # shift data 
    def shift_data(data, n):
        data_shift = data.shift(n)
        colnames = [fea + '_s' + str(n) for fea in data.columns]
        data_shift.columns = colnames
        return data_shift

    # percent change for numerical feature
    def pctchg_data(data):
        data_pctchg = data.pct_change()
        colnames = [fea + '_pctchg' for fea in data.columns]
        data_pctchg.columns = colnames
        return data_pctchg

    df_shift1 = shift_data(data2[num_features+cat_features], 1)
    df_shift2 = shift_data(data2[num_features+cat_features], 2)
    df_shift3 = shift_data(data2[num_features+cat_features], 3)
    df_pctchg = pctchg_data(data2[num_features])
    df_pctchg_shift1 = shift_data(df_pctchg, 1)
    df_pctchg_shift2 = shift_data(df_pctchg, 2)
    df_pctchg_shift3 = shift_data(df_pctchg, 3)

    df_all = pd.concat([data2,
            df_shift1,
            df_shift2,
            df_shift3,
            df_pctchg,
            df_pctchg_shift1,
            df_pctchg_shift2,
            df_pctchg_shift3], axis = 1)
    
    df_all.index = pd.to_datetime(df_all.index)
    
    return df_all

def data_slice(data, date_start):
    '''
    this function is to select data from date_start and to fill in empty data
    '''
    data2 = data[data.index >= date_start]
    data2= data2.bfill().ffill()
    data2 = data2.replace(np.Inf,0).replace(-np.Inf, 0)
    return data2

def feature_selection_lasso(data, target, alpha_):
    '''
    target: the target column name
    alpha_: the parameter of lasso regression, the extent of penalty
    '''
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X = data.drop(target, axis=1)
    y = data[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha_)
    lasso.fit(X_scaled, y)

    selected_features = list(X.columns[(lasso.coef_ != 0)])
    # remove sentiment score without shift to avoid future leak
    selected_features.remove('Sentiment')
    selected_features.remove('sentiment_score_LM')

    print("Selected Features:", selected_features)
    print("length of Selected Features:", len(selected_features))
    return selected_features

def rolling_randomforest(X, y, base_months, best_para, random_seed=42):
    '''
    this function is used to only predict value of next month by using randomforest model
    each time add one month true value into train dataset
    the initial window size is base_months
    '''
    from sklearn.preprocessing import StandardScaler
    # define train window
    start_train = X.index.min()
    end_train = start_train + pd.DateOffset(months = base_months)
    print(start_train, end_train)
    end_test = X.index.max()

    # store predictions and true values
    predictions = []
    true_values = []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    from sklearn.ensemble import RandomForestClassifier

    # rolling window to train and predict next 1 month
    while end_train <= end_test:
        # slice dataset
        train_mask = (X.index < end_train)
        test_mask = (X.index >= end_train) & (X.index < end_train + pd.DateOffset(months=1))
        X_train, y_train = X_scaled[train_mask], y[train_mask].values.ravel()
        X_test, y_test = X_scaled[test_mask], y[test_mask].values.ravel()

        # train randomforest model
        rf_model = RandomForestClassifier(random_state=random_seed, **best_para)
        rf_model.fit(X_train, y_train)

        # predict next month
        y_pred = rf_model.predict(X_test)

        # store predictions and true values
        predictions.extend(y_pred)
        true_values.extend(y_test)

        # move window by 1 month
        end_train += pd.DateOffset(months=1)

        # update the last model
        last_rf_model = rf_model

    return true_values, predictions, last_rf_model


    
