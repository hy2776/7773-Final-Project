"""

Simple stand-alone script showing end-to-end training of a regression model using Metaflow. 
This script ports the composable script into an explicit dependency graphg (using Metaflow syntax)
and highlights the advantages of doing so. This script has been created for pedagogical purposes, 
and it does NOT necessarely reflect all best practices.

Please refer to the slides and our discussion for further context.

MAKE SURE TO RUN THIS WITH METAFLOW LOCAL FIRST

"""
from comet_ml import Experiment
from metaflow import FlowSpec, step, Parameter, IncludeFile, current
from datetime import datetime
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# make sure we are running locally for this
assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'


class Multiclass_RandomForest(FlowSpec):
    """
    SampleRegressionFlow is a minimal DAG showcasing reading data from a file 
    and training a model successfully.
    """
    
    # if a static file is part of the flow, 
    # it can be called in any downstream process,
    # gets versioned etc.
    # https://docs.metaflow.org/metaflow/data#data-in-local-files
    DATA_FILE = IncludeFile(
        'dataset',
        help='Text file with the dataset',
        is_text=True,
        default='../data/preprocessed/dataset_merged_with_sentiment.csv'
    )


    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        # debug printing - this is from https://docs.metaflow.org/metaflow/tagging
        # to show how information about the current run can be accessed programmatically
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self): 
        """
        Read the data in from the static file
        """
        from io import StringIO
        #Load datasets from the IncludeFile object
        self.df = pd.read_csv(StringIO(self.DATA_FILE))
        self.df.set_index('observation_date', inplace = True)
        print(f"dataset has {self.df.shape[0]} row and {self.df.shape[1]} columns.")
        self.next(self.check_data)
    
    # we need to add checks
    @step
    def check_data(self):
        """
        check labels and outliers
        """
        from Flow_utils import replace_outliers_with_boundaries
        self.target = ['Target_category']
        self.num_features = ['PCE','Consumer_Sentiment','Supply_NewHouse','CPI','ISM PMI','Housing_starts','Unemp_rate','SPX','10y_try','sentiment_score_LM']
        self.cat_features = ['Sentiment']

        # define data that needs to be checked, drop the first line
        df_check = self.df[self.num_features+self.target+self.cat_features].drop(self.df.index[0])
        # check we actually have only 3 target classes, as expected 
        all_labels = set(df_check[self.target].values.ravel())
        assert len(all_labels) == 3
        print("All labels are: {}".format(all_labels))

        # check numerical features (except sentiment_score_LM) outliers and replace with boundaries
        self.df = replace_outliers_with_boundaries(self.df, self.num_features, multiplier=3)
        # if data is all good, let's go to training
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        """
        Check data is ok before training starts
        """
        from Flow_utils import feature_preprocess, data_slice, feature_selection_lasso

        # generate shift and percent change feature
        self.df_all = feature_preprocess(self.df, self.target, self.num_features, self.cat_features)
        print(self.df_all.shape)
        # select data from 1994-01-01
        self.df_all2 = data_slice(self.df_all, '1994-01-01')
        print(self.df_all2.shape)
        # select features by using Lasso regression
        self.selected_features = feature_selection_lasso(self.df_all2, self.target, 0.05)

        self.next(self.prepare_train_and_test_dataset)

    @step
    def prepare_train_and_test_dataset(self):
        from sklearn.preprocessing import StandardScaler
        from imblearn.over_sampling import SMOTE

        # Assign values to the X and y variables:
        self.X = self.df_all2[self.selected_features]
        self.y = self.df_all2[self.target]
        
        # Time Series splitting
        self.X_train = self.X.iloc[:280].values
        self.X_test = self.X.iloc[280:].values
        print(self.X_test.shape)

        self.y_train = self.y.iloc[:280].values.ravel()
        self.y_test = self.y.iloc[280:].values.ravel()
        print(self.y_train.shape, self.y_test.shape)

        # oversample data
        # oversample = SMOTE(k_neighbors=3)
        # self.X_train, self.y_train = oversample.fit_resample(self.X_train, self.y_train)

        # scale data
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
       
        self.next(self.train_model_baseline)

    @step
    def train_model_baseline(self):
        """
        Train a hyper tune Random forest on the training set
        """
        from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.model_selection import cross_val_score

        # define a random seed
        self.random_seed = 666
        # define a timeseries split
        tscv = TimeSeriesSplit(n_splits=5)

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Class weight
        class_weight = ['balanced', 'balanced_subsample', None]
        # Create the random grid
        random_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap,
            'class_weight': class_weight
        }
        
        # Create a random forest classifier
        rf = RandomForestClassifier()

        # Use random search to find the best hyperparameters
        rand_search = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                                    n_iter = 5, cv = tscv, verbose=2, random_state=self.random_seed, n_jobs = -1)

        # Fit the random search object to the data
        rand_search.fit(self.X_train_scaled, self.y_train)
        # Create a variable for the best model
        self.best_rf = rand_search.best_estimator_
        self.best_para = rand_search.best_params_
        print(self.best_para)
        # evaluate a model
        def evaluate_model(X, y, model):
            # define evaluation procedure
            tscv = TimeSeriesSplit(n_splits=5)
            # evaluate model
            scores = cross_val_score(model, X, y, scoring='accuracy', cv=tscv, n_jobs=-1)
            print(scores)
            return scores
        
        scores = evaluate_model(self.X_train_scaled, self.y_train, self.best_rf)
        # summarize performance
        print('Validation Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        self.valid_acc = np.mean(scores)   

        # Generate predictions with the best model
        self.y_pred = self.best_rf.predict(self.X_test_scaled)
        print(self.X_test_scaled.shape)

        self.next(self.train_model_rolling)

    @step
    def train_model_rolling(self):
        """
        train on rolling window with the best model's parameter
        """
        from Flow_utils import rolling_randomforest
        self.y_test_rolling, self.y_pred_rolling, self.last_rf_model = rolling_randomforest(self.X, self.y, 280, self.best_para, self.random_seed)
        self.next(self.model_evaluation)

    @step 
    def model_evaluation(self):
        """
        evaluate models and show feature importance
        """
        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
        
        # Organize model outputs in a dictionary
        models = {
            'baseline': {
                'y_test': self.y_test,
                'y_pred': self.y_pred,
                'model': self.best_rf
            },
            'rolling': {
                'y_test': self.y_test_rolling,
                'y_pred': self.y_pred_rolling,
                'model': self.last_rf_model
            }
        }
         
        for model_name, data in models.items():
             # comet log
            exp = Experiment(
                api_key = 'dJsuCqjZCBbm3ZPHtC6FKNk9N',
                project_name = "hy2776_project",
                workspace = "nyu-fre-7773-2021",
                auto_param_logging = False
            )
            accuracy = accuracy_score(data['y_test'], data['y_pred'])
            precision = precision_score(data['y_test'], data['y_pred'], average='weighted')
            recall = recall_score(data['y_test'], data['y_pred'], average='weighted')
            f1 = f1_score(data['y_test'], data['y_pred'], average='weighted')
            cm = confusion_matrix(data['y_test'], data['y_pred'])

            # Store metrics in Comet
            exp.log_metrics({
                f'{model_name}_test_accuracy': accuracy,
                f'{model_name}_test_precision': precision,
                f'{model_name}_test_recall': recall,
                f'{model_name}_test_f1_score': f1
            })

            # Log confusion matrix
            exp.log_confusion_matrix(
                labels=sorted(np.unique(data['y_test'])), 
                matrix=cm, 
                title=f'{model_name} confusion matrix')
      
            exp.log_parameters(self.best_para)
            exp.end()

            # Log feature importance
            feature_importances = data['model'].feature_importances_
            sorted_idx = np.argsort(feature_importances)[::-1]
            sorted_feature_names = [self.selected_features[i] for i in sorted_idx]

            print(f"Top 10 Features in {model_name} Model:")
            for i in range(10):
                print(f"{sorted_feature_names[i]}: {feature_importances[sorted_idx[i]]}")
            print("\n")
            
        self.next(self.end)

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    Multiclass_RandomForest()
