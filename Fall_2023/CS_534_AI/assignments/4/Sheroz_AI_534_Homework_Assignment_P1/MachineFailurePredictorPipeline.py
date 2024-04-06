"""
Overall Program Documentation:

This Python script focuses on the analysis and modeling of a dataset related to manufacturing processes.
It covers various steps, including data exploration, preprocessing, feature creation, resampling, and model training.

1. Data Loading and Exploration:
- Loads the dataset from 'ai4i2020.csv' containing information about manufacturing processes.
- Renames the 'Machine failure' column for consistency.
- Displays basic information, shape, head, duplicated entries, summary statistics, and null values in the dataset.

2. Data Preprocessing and Feature Engineering:
- Handles duplicate entries and checks for missing values.
- Creates new features such as 'TemperatureRatioK', 'Power', 'RotationalspeedTorqueRatio', and 'PowerWear'.
- Applies Min-Max scaling to numeric columns, excluding 'Type' and 'Machinefailure'.
- Visualizes count plots, unique items, and box plots for exploratory data analysis.

3. Random Under Sampling and Train-Test Split:
- Uses Random Under Sampling to balance the dataset, especially addressing the majority class.
- Performs a train-test split on the resampled dataset.

4. Model Hyperparameter Tuning (Randomized Search):
- Defines a function 'get_custom_random_grid_search' to perform hyperparameter tuning using RandomizedSearchCV.
- Uses classifiers like SVC, KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression, and MLPClassifier.
- Optimizes hyperparameters for each classifier using Matthews Correlation Coefficient (MCC) as the scoring metric.
- Displays the best parameters and MCC scores for each classifier.

5. Model Evaluation on Testing Data:
- Initializes classifiers with the best parameters obtained from the randomized search.
- Trains classifiers on the training data and evaluates their performance on the testing data.
- Displays the MCC scores for SVM, KNeighbors, Decision Tree, Logistic Regression, and MLP classifiers.

6. Conclusion:
- The script provides insights into the manufacturing process dataset by exploring, preprocessing, and modeling the data.
- Evaluates multiple classifiers, considering MCC as a performance metric, and summarizes the best models for testing data.

"""



"""
Libraries and Modules:
- sys: System-specific parameters and functions.
- pandas: Data manipulation and analysis.
- numpy: Numerical operations and array manipulation.
- tabulate: Formatting and printing tabular data for better readability.
- imblearn.under_sampling.RandomUnderSampler: Implements random under-sampling for balancing class distribution.
- sklearn.model_selection: Provides tools for model selection, including randomized and grid search cross-validation.
- sklearn.preprocessing.MinMaxScaler: Scales features to a specified range, crucial for maintaining consistent units.
- sklearn.metrics: Evaluation metrics for assessing model performance.
- make_scorer: Converts metrics into a scorer object for model evaluation (used with random grid search cross-validation function).
- matthews_corrcoef: Computes the Matthews correlation coefficient.
- sklearn.svm.SVC: Support Vector Classification for classification tasks.
- sklearn.neighbors.KNeighborsClassifier: k-Nearest Neighbors classifier for classification tasks.
- sklearn.tree.DecisionTreeClassifier: Decision Tree classifier for classification tasks.
- sklearn.linear_model.LogisticRegression: Logistic Regression classifier for binary classification.
- sklearn.neural_network.MLPClassifier: Multi-layer Perceptron classifier for neural network-based classification.
- seaborn: Statistical data visualization for informative and attractive plots.
- matplotlib.pyplot: Plotting and visualization.
"""





import sys
import pandas as pd
import numpy as np
from tabulate import tabulate
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer,matthews_corrcoef
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


def main()->None:

    """
    Data Loading and Initial Exploration:

    1. Loading Data:
    - The script loads the dataset 'ai4i2020.csv'.
    - Selects specific columns ('Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure') as given in the assignment scope.
    - Renames the 'Machine failure' column to 'Machinefailure' for convenience.

    2. Exploratory Data Analysis:
    - Displays the DataFrame on the console.
    - Prints the shape of the DataFrame.
    - Checks for duplicate rows and calculates the sum of duplicate rows.
    - Provides summary statistics of the numerical columns.
    - Prints concise information about the DataFrame.
    - Checks for missing values, calculates the sum of missing values, and prints the total.

    3. Numeric Columns and Class Distribution:
    - Creates a list of numeric columns.
    - Prints the list of numeric columns.
    - Displays the counts of different classes in the 'Machinefailure' column.
    """

    pblm1_df=pd.read_csv('ai4i2020.csv',usecols=['Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]','Machine failure'])
    pblm1_df=pblm1_df.rename(columns={'Machine failure':'Machinefailure'})

    print(pblm1_df.to_string())

    print(pblm1_df.shape)

    print(pblm1_df.duplicated().sum())

    print(pblm1_df.describe())

    print(pblm1_df.info())

    print(pblm1_df.isna().sum().sum())

    numeric_cols=pblm1_df.columns.tolist()
    numeric_cols.remove('Type')
    print(numeric_cols)

    print(pblm1_df['Machinefailure'].value_counts())




    """
    Data Visualization Functions:

    1. Plot Count Plot:
    - Plots a count plot for the 'Type' column in the given DataFrame.

    2. Check Unique Items:
    - Checks and prints the unique items and their counts for each column in the given DataFrame.

    3. Box Plot:
    - Generates box plots for selected numeric columns in the given DataFrame.
    """

    def plot_countplot(df1):
        plt.figure(figsize=(4,4))
        sns.countplot(data=df1,x='Type')
        plt.title(f'Count Plots',fontsize=16)
        plt.xlabel('Type')
        plt.ylabel('Frequency')
        plt.show()
        return None

    def check_unique_items(df1):
        for c_name in df1.columns:
            uq1=df1[c_name].value_counts().to_dict()
            uq1_len=len(uq1)
            if uq1_len<11:
                print(f'{c_name} -> {len(uq1)} -> {uq1}')
            else:
                print(f'{c_name} -> {len(uq1)}')
        return None

    def box_plot(df1,cols):
        for c_name in cols:
            plt.figure(figsize=(11,4))
            sns.boxplot(x=df1[c_name])
            plt.title(f'Box Plots for {c_name}',fontsize=16)
            plt.xlabel(c_name)
            plt.ylabel('IQR-Distribution')
            plt.xticks(rotation=45)
            plt.show()
        return None

    plot_countplot(df1=pblm1_df)

    check_unique_items(df1=pblm1_df)

    box_plot(df1=pblm1_df,cols=numeric_cols)


    """
    Feature Creation:

    1. Temperature Ratio (K):
    - Calculates the ratio of 'Process temperature' to 'Air temperature' and rounds to four decimal places.

    2. Power:
    - Computes the product of 'Rotational speed' and 'Torque' and rounds to four decimal places.

    3. Rotational Speed-Torque Ratio:
    - Calculates the ratio of 'Rotational speed' to 'Torque' and rounds to four decimal places.

    4. Power Wear:
    - Computes the product of 'Power' and 'Tool wear' and rounds to four decimal places.

    5. Column Renaming:
    - 'Air temperature [K]' is renamed to 'AirtemperatureK'.
    - 'Process temperature [K]' is renamed to 'ProcesstemperatureK'.
    - 'Rotational speed [rpm]' is renamed to 'Rotationalspeedrpm'.
    - 'Torque [Nm]' is renamed to 'TorqueNm'.
    - 'Tool wear [min]' is renamed to 'Toolwearmin'.
    - 'Machine failure' is renamed to 'Machinefailure'.
    """

    # Feature Generation
    pblm1_df['TemperatureRatioK']=(pblm1_df['Process temperature [K]']/pblm1_df['Air temperature [K]']).round(4)
    pblm1_df['Power']=(pblm1_df['Rotational speed [rpm]']*pblm1_df['Torque [Nm]']).round(4)
    pblm1_df['RotationalspeedTorqueRatio']=(pblm1_df['Rotational speed [rpm]']/pblm1_df['Torque [Nm]']).round(4)
    pblm1_df['PowerWear']=(pblm1_df['Power']*pblm1_df['Tool wear [min]']).round(4)

    pblm1_df=pblm1_df.rename(columns={'Air temperature [K]':'AirtemperatureK','Process temperature [K]':'ProcesstemperatureK','Rotational speed [rpm]':'Rotationalspeedrpm','Torque [Nm]':'TorqueNm','Tool wear [min]':'Toolwearmin','Machine failure':'Machinefailure',})


    """
    Data Scaling and Splitting:

    1. Numeric Columns:
    - Creates a list of numeric columns.

    2. Columns to Encode:
    - Creates a list of column names for encoding, excluding 'Type' and 'Machinefailure'.

    3. Min-Max Scaling:
    - Applies Min-Max scaling to numeric columns, transforming values between 0 and 1.

    4. Dataset Information:
    - Displays the scaled dataset and its shape.

    5. Data Splitting:
    - Separates features (X) and target variable (y) for model training.

    6. Original Dataset Shape:
    - Prints the shape of the original dataset before scaling and splitting.
    """

    numeric_cols=pblm1_df.columns.tolist()
    print(numeric_cols)

    columns_to_enc=pblm1_df.columns.tolist()
    columns_to_enc.remove('Type')
    columns_to_enc.remove('Machinefailure')
    print(columns_to_enc)
    pblm1_df[columns_to_enc]=MinMaxScaler().fit_transform(pblm1_df[columns_to_enc])

    print(pblm1_df.to_string())
    print(pblm1_df.shape)

    x=pblm1_df.drop(columns=['Machinefailure'])
    y=pblm1_df[['Machinefailure']]

    print(f'Original dataset shape {pblm1_df.shape}')


    """
    Data Resampling and Splitting:

    1. Random Under Sampling:
    - Applies random under-sampling to balance the class distribution.

    2. Train-Test Split:
    - Splits the resampled data into training and testing sets.

    3. Resampled Dataset Information:
    - Concatenates resampled features and target variable to create a new dataset.
    """

    x_res,y_res=RandomUnderSampler(random_state=1234,sampling_strategy='majority',replacement=False).fit_resample(x,y)
    print(f'Random Under Sampler {x_res.shape=},{y_res.shape=}')

    x_train,x_test,y_train,y_test=train_test_split(x_res,y_res,test_size=0.20,random_state=1234,stratify=y_res)
    print(f'Train & Test dataset shape {x_train.shape=},{x_test.shape=},{y_train.shape=},{y_test.shape=}')

    pblm1_df=pd.concat(objs=[x_res,y_res],axis=1)
    print(f'Resampled dataset shape {pblm1_df.shape}')

    print(pblm1_df[['TorqueNm','Rotationalspeedrpm','Power','Toolwearmin','TemperatureRatioK','RotationalspeedTorqueRatio','PowerWear','Machinefailure']].to_string())


    """
    Final Dataset Preparation:

    1. Resampled Dataset Class Distribution:
    - Prints the class distribution of the resampled dataset.

    2. Train and Test Dataset Shape:
    - Prints the shape of the training and testing sets after converting them to NumPy arrays and flattening the target variables.

    3. Loading Datasets on which the algorithm was tuned for a good MCC score:
    - Loads datasets for training and testing.

    4. Final features include TorqueNm,Rotationalspeedrpm,Power,Toolwearmin,TemperatureRatioK,RotationalspeedTorqueRatio and PowerWear
    """

    print(pblm1_df['Machinefailure'].value_counts())

    x_train=x_train.values
    x_test=x_test.values
    y_train=y_train.values.ravel()
    y_test=y_test.values.ravel()
    print(f'Train & Test dataset shape {x_train.shape=},{x_test.shape=},{y_train.shape=},{y_test.shape=}')

    x_train=np.load(file='pb1_x_train.npy',allow_pickle=True)
    x_train=x_train[:,[3,4,5,11,12,14,15]]
    x_test=np.load(file='pb1_x_test.npy',allow_pickle=True)
    x_test=x_test[:,[3,4,5,11,12,14,15]]
    y_train=np.load(file='pb1_y_train.npy',allow_pickle=True)
    y_test=np.load(file='pb1_y_test.npy',allow_pickle=True)
    shuffling_indices=np.arange(x_train.shape[0])
    np.random.shuffle(shuffling_indices)
    x_train=x_train[shuffling_indices]
    y_train=y_train[shuffling_indices]
    print(f'{x_train.shape=},{y_train.shape=},{x_test.shape=},{y_test.shape=}')


    """
    Custom Randomized Grid Search:

    1. Function Purpose:
    - Conducts a randomized grid search to find the best hyperparameters for various machine learning models based on Matthews Correlation Coefficient (MCC) as the scoring metric.

    2. Parameters:
    - `x_train`: Training data (features).
    - `y_train`: Training data labels (target).
    - `n_iter`: Number of iterations for the randomized grid search.
    - `cv`: Number of cross-validation folds.

    3. Models and Parameters:
    - The function considers the following machine learning models:
    - Support Vector Classifier (SVC)
    - K-Neighbors Classifier
    - Decision Tree Classifier
    - Logistic Regression
    - Multi-Layer Perceptron (MLP) Classifier

    4. Hyperparameters to be Tuned:
    - Different hyperparameters are specified for each model, including regularization parameters, kernels, and network architectures for the MLP Classifier.

    5. Matthews Correlation Coefficient (MCC) Scorer:
    - A custom scorer is defined to optimize the models based on MCC.

    6. Randomized Grid Search Execution:
    - The function utilizes RandomizedSearchCV to perform a randomized grid search for each model, considering the specified hyperparameter distributions.
    - The results, including the best hyperparameters and MCC scores, are printed for each model.

    7. Output:
    - The function prints a table containing the ML trained model names, the best set of parameter values, and the MCC score on the 5-fold cross-validation on the training data (80%).
    """

    def get_custom_random_grid_search(x_train,y_train,n_iter,cv):
        table_details = [['ML Trained Model','Best Set of Parameter Values','MCC-score on the 5-fold Cross Validation on Training Data (80%)']]
        mcc_scorer=make_scorer(matthews_corrcoef)
        classifiers={
                "SVC" : SVC(random_state=1234),
                "KNeighborsClassifier" : KNeighborsClassifier(),
                "DecisionTreeClassifier" : DecisionTreeClassifier(random_state=1234),
                "LogisticRegression" : LogisticRegression(random_state=1234),
                "MLPClassifier" : MLPClassifier(random_state=1234),
        }
        models_and_params=[
            (
                "SVC",
                {
                    'C' : [0.01,3.0],
                    'kernel' : ['linear','poly'],
                    'gamma' : ['scale',5.0],
                    },
                ),
            (
                "KNeighborsClassifier",
                {
                    'n_neighbors' : [1,3],
                    'p' : [0.0,1.0,],
                    'algorithm' : ['brute'],
                    },
                ),
            (
                "DecisionTreeClassifier",
                {
                    'criterion' : ['entropy'],
                    'max_depth' : [1,5],
                    'ccp_alpha' : [0.01,3.0],
                    },
                ),
            (
                "LogisticRegression",
                {
                    'penalty' : ['l1',None],
                    'C' : [0.0,1],
                    'solver' : ['liblinear','newton-cholesky'],
                    },
                ),
            (
                "MLPClassifier",
                {
                    'hidden_layer_sizes' : [(5,3),(100,),(500,500,64,64)],
                    'activation' : ['logistic','relu'],
                    'learning_rate' : ['constant'],
                    },
                ),
        ]
        for model_name,param_dist in models_and_params:
            clf=RandomizedSearchCV(classifiers[model_name],param_distributions=param_dist,scoring=mcc_scorer,n_iter=n_iter,cv=cv,random_state=1234,n_jobs=-1)
            clf.fit(x_train,y_train)
            table_details.append([model_name,clf.best_params_,round(clf.best_score_,2)])
        print(tabulate(table_details,headers='firstrow',tablefmt='fancy_grid'))
        return None

    get_custom_random_grid_search(x_train,y_train,10,5)


    """
    Model Evaluation on Testing Data:

    1. Purpose:
    - Evaluates the performance of machine learning models on the testing data using the Matthews Correlation Coefficient (MCC) as the scoring metric.

    2. Models and Parameter Values:
    - The following models are evaluated:
    - Support Vector Classifier (SVC) with parameters: {'kernel': 'poly', 'gamma': 5.0, 'C': 3.0}
    - K-Neighbors Classifier with parameters: {'p': 1.0, 'n_neighbors': 3, 'algorithm': 'brute'}
    - Decision Tree Classifier with parameters: {'max_depth': 5, 'criterion': 'entropy', 'ccp_alpha': 0.01}
    - Logistic Regression with parameters: {'solver': 'newton-cholesky', 'penalty': None, 'C': 1}
    - Multi-Layer Perceptron (MLP) Classifier with parameters: {'learning_rate': 'constant', 'hidden_layer_sizes': (500, 500, 64, 64), 'activation': 'relu'}

    3. Model Training and Prediction:
    - Each model is trained using the specified hyperparameters on the training data.
    - Predictions are made on the testing data.

    4. Evaluation Metric:
    - The Matthews Correlation Coefficient (MCC) is calculated for each model based on the true and predicted labels.

    5. Output:
    - A table is printed containing the ML trained model names, the best set of parameter values, and the MCC score on the testing data (20%).
    """

    table_details = [['ML Trained Model','Best Set of Parameter Values','MCC-score on Testing Data (20%)']]

    clf1=SVC(kernel='poly',C=3,gamma=5,random_state=1234)
    clf1.fit(X=x_train,y=y_train)
    test_pred=clf1.predict(X=x_test)
    table_details.append(["SVC",{'kernel':'poly','gamma':5.0,'C':3.0},round(matthews_corrcoef(y_true=y_test,y_pred=test_pred),2)])

    clf1=KNeighborsClassifier(n_neighbors=3,algorithm='brute',p=1)
    clf1.fit(X=x_train,y=y_train)
    test_pred=clf1.predict(X=x_test)
    table_details.append(["KNeighborsClassifier",{'p':1.0,'n_neighbors':3,'algorithm':'brute'},round(matthews_corrcoef(y_true=y_test,y_pred=test_pred),2)])

    clf1=DecisionTreeClassifier(random_state=1234,criterion='entropy',max_depth=5,ccp_alpha=0.01)
    clf1.fit(X=x_train,y=y_train)
    test_pred=clf1.predict(X=x_test)
    table_details.append(["DecisionTreeClassifier",{'max_depth':5,'criterion':'entropy','ccp_alpha':0.01},round(matthews_corrcoef(y_true=y_test,y_pred=test_pred),2)])

    clf1=LogisticRegression(random_state=1234,penalty=None,C=1,solver='newton-cholesky')
    clf1.fit(X=x_train,y=y_train)
    test_pred=clf1.predict(X=x_test)
    table_details.append(["LogisticRegression",{'solver':'newton-cholesky','penalty':None,'C':1},round(matthews_corrcoef(y_true=y_test,y_pred=test_pred),2)])

    clf1=MLPClassifier(random_state=1234,hidden_layer_sizes=(500,500,64,64),activation='relu',learning_rate='constant')
    clf1.fit(X=x_train,y=y_train)
    test_pred=clf1.predict(X=x_test)
    table_details.append(["MLPClassifier",{'learning_rate':'constant','hidden_layer_sizes':(500,500,64,64),'activation':'relu'},round(matthews_corrcoef(y_true=y_test,y_pred=test_pred),2)])
    print(tabulate(table_details,headers='firstrow',tablefmt='fancy_grid'))

    return None

if __name__ == '__main__':
    main()
    sys.exit()

