import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint, loguniform
import warnings
import time


class ML_Models():
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.X_train_vectorized = None
        self.X_test_vectorized = None
        self.y_train = None
        self.y_test = None
        
    def run(self, ml_df):
        X = ml_df['text']
        y = ml_df['label_ids']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Vectorizing text data
        vectorizer = TfidfVectorizer()
        X_train_vectorized = vectorizer.fit_transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
        
        self.X_train_vectorized = X_train_vectorized
        self.X_test_vectorized = X_test_vectorized

        perform_list = []

        def run_model(model_name):
            mdl = ''
            if model_name == 'Logistic Regression':
                mdl = LogisticRegression()
            elif model_name == 'Random Forest':
                mdl = RandomForestClassifier()
            elif model_name == 'Naive Bayes':
                mdl = MultinomialNB(alpha=1.0, fit_prior=True)
            elif model_name == 'Support Vector Classifer':
                mdl = SVC()
            elif model_name == 'Decision Tree Classifier':
                mdl = DecisionTreeClassifier()

            mdl.fit(X_train_vectorized, y_train)
            y_pred = mdl.predict(X_test_vectorized)

            # Performance metrics
            accuracy = round(accuracy_score(y_test, y_pred), 2)
            # Get precision, recall, f1 scores
            precision, recall, f1score, support = score(y_test, y_pred, average='weighted')
            print(f"{model_name}\n{classification_report(y_test, y_pred)}\n")

            # Add performance parameters to list
            perform_list.append(dict([
                ('Model', model_name),
                ('Test Accuracy', round(accuracy, 2)),
                ('Precision', round(precision, 2)),
                ('Recall', round(recall, 2)),
                ('F1', round(f1score, 2))]))

        model_names = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 
                       'Support Vector Classifer','Decision Tree Classifier']

        for m in model_names:
            run_model(m)

        # Create Dataframe of Model, Accuracy, Precision, Recall, and F1
        model_performance = pd.DataFrame(data=perform_list)
        model_performance = model_performance[['Model', 'Test Accuracy', 'Precision', 'Recall', 'F1']]
        model_performance = model_performance.set_index('Model')
        return model_performance

    def plot_performance(self, model_performance):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))

        model_performance['Test Accuracy'].sort_values(ascending=False).plot(kind='bar', ax=ax[0,0], rot=45, title="Test accuracy", xlabel="")
        model_performance['Precision'].sort_values(ascending=False).plot(kind='bar', ax=ax[0,1], rot=45, title="Precision", xlabel="")
        model_performance['Recall'].sort_values(ascending=False).plot(kind='bar', ax=ax[1,0], rot=45, title="Recall", xlabel="")
        model_performance['F1'].sort_values(ascending=False).plot(kind='bar', ax=ax[1,1], rot=45, title="F1", xlabel="")
        fig.tight_layout()
        

        
    def fine_tune(self, model_name: str, use_all_CPUs: bool, 
                  number_of_iterations: int, num_cv: int):
        """
        FINE TUNES ONLY 2 MODELS: EITHER LOGISTIG REGRESSION "logreg" OR SVC "svc" OR "dt".
        TO FINE TUNE ON PARTICULAR DATASET THE METHOD "RUN" SHOULD BE CALLED PRIOR TO FINETUNING

        Args:
            model_name (str): _description_
            use_all_CPUs (bool): _description_
            number_of_iterations (int): _description_
            num_cv (int): _description_
        """
        
        def convert_time(seconds):
            seconds = seconds % (24 * 3600)
            hour = seconds // 3600
            seconds %= 3600
            minutes = seconds // 60
            seconds %= 60
            return "%d:%02d:%02d" % (hour, minutes, seconds)
        
        if model_name == "logreg":
            # Track runtime
            start_t = time.time()

            with warnings.catch_warnings():
                # Ignore warnings while running this chunk of code
                warnings.simplefilter("ignore")
                
                # Define hyperparameters grid for Logistic Regression
                param_dist_lr = {
                    'C': loguniform(1e-5, 100),  # Continuous loguniform distribution between 0.00001 and 100
                    'penalty': ['none', 'l1', 'l2', 'elasticnet'], # Define different penalties
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],  # Define different solvers
                    'max_iter': randint(100, 1000) # Define maximum iterations in our LogReg model
                }
                
                # Instantiate Logistic Regression classifier
                lr = LogisticRegression()
                
                # Perform random search
                if use_all_CPUs:
                    random_search_lr = RandomizedSearchCV(estimator=lr, param_distributions=param_dist_lr, n_iter=number_of_iterations, cv=num_cv, scoring='accuracy', random_state=123, n_jobs=-1)
                else:
                    random_search_lr = RandomizedSearchCV(estimator=lr, param_distributions=param_dist_lr, n_iter=number_of_iterations, cv=num_cv, scoring='accuracy', random_state=123)
                random_search_lr.fit(self.X_train_vectorized, self.y_train)
                
                # Get best parameters and best scores
                best_params_lr = random_search_lr.best_params_
                best_accuracy_lr = random_search_lr.best_score_
                
                print("Best Parameters for Logistic Regression:", best_params_lr)
                print("Best Accuracy Score for Logistic Regression:", best_accuracy_lr)

            end_t = time.time()
            runtime = end_t - start_t
            print(f"Total runtime: {convert_time(runtime)}")
            return best_params_lr
        
        elif model_name == "svc":
            # Define hyperparameters grid for SVC
            start_t = time.time()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                param_dist_svc = {
                    'C': loguniform(1e-5, 100),  # Continuous uniform distribution between 0.1 and 100
                    'gamma': ['scale', 'auto'],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'degree': randint(1, 10)  # Additional parameter for SVC
                }
                
                # Instantiate SVC classifier
                svc = SVC()
                
                # Perform random search
                if use_all_CPUs:
                    random_search_svc = RandomizedSearchCV(estimator=svc, param_distributions=param_dist_svc, n_iter=number_of_iterations, 
                                                           cv=num_cv, scoring='accuracy', random_state=123, n_jobs=-1)
                else:
                    random_search_svc = RandomizedSearchCV(estimator=svc, param_distributions=param_dist_svc, n_iter=number_of_iterations, 
                                                           cv=num_cv, scoring='accuracy', random_state=123)
                random_search_svc.fit(self.X_train_vectorized, self.y_train)
                
                # Get best parameters and best scores
                best_params_svc = random_search_svc.best_params_
                best_accuracy_svc = random_search_svc.best_score_
                
                print("Best Parameters for SVC:", best_params_svc)
                print("Best Accuracy Score for SVC:", best_accuracy_svc)

            end_t = time.time()
            runtime = end_t - start_t
            print(f"Total runtime: {convert_time(runtime)}")
            return best_params_svc
        
        elif model_name == "dt":

            start_t = time.time()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                param_dist_dt = {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': randint(1, 50),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 20)
                }

                # Instantiate Decision Tree classifier
                dt = DecisionTreeClassifier()

                # Perform random search
                if use_all_CPUs:
                    random_search_dt = RandomizedSearchCV(estimator=dt, param_distributions=param_dist_dt, n_iter=number_of_iterations,
                                                        cv=num_cv, scoring='accuracy', random_state=123, n_jobs=-1)
                else:
                    random_search_dt = RandomizedSearchCV(estimator=dt, param_distributions=param_dist_dt, n_iter=number_of_iterations,
                                                        cv=num_cv, scoring='accuracy', random_state=123)
                random_search_dt.fit(self.X_train_vectorized, self.y_train)

                # Get best parameters and best scores
                best_params_dt = random_search_dt.best_params_
                best_accuracy_dt = random_search_dt.best_score_

                print("Best Parameters for Decision Tree:", best_params_dt)
                print("Best Accuracy Score for Decision Tree:", best_accuracy_dt)

            end_t = time.time()
            runtime = end_t - start_t
            print(f"Total runtime: {runtime} seconds")

            return best_params_dt
                        
        else:
            print("Model name not defined!")
            return
            

