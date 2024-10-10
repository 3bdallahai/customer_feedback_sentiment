import mlflow
from mlflow_utils import create_mlflow_experiment, get_mlflow_experiment
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier




if __name__ == "__main__":
    experiment = get_mlflow_experiment(experiment_name="feddback classification")

    with mlflow.start_run(run_name="run 2",experiment_id=experiment.experiment_id)as run:

        balanced_df = pd.read_csv("balanced_df.csv")
        balanced_df.dropna(inplace=True)
        X  = balanced_df['review']
        Y = balanced_df['review-label']-1

        vectorizer = TfidfVectorizer()
        vectorizer.fit(X)
        X = vectorizer.transform(X)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)

        x_train_np = x_train.to_numpy() if isinstance(x_train, pd.Series) else x_train
        y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train



        mlflow.autolog()
        model = VotingClassifier(estimators=[
            ('svc',SVC(probability=True)),
            ('lr', LogisticRegression(max_iter=500)), 
            ('xgb', XGBClassifier())], voting='soft')

        model.fit(x_train_np, y_train_np)

        # with open("metrics/classification_metrics.txt","w") as f:

        #     f.write(f"Train set report:\n {classification_report(y_train, train_pred)}\n") 
        #     f.write(f"Test set report:\n { classification_report(y_test, test_pred)}\n")


        # mlflow.log_artifact("metrics/classification_metrics.txt")


        # fig_conf_matrix = plt.figure()
        # conf_matrix_display = ConfusionMatrixDisplay.from_predictions(y_test, test_pred, ax=plt.gca())
        # plt.title("confusion matrix")
        # mlflow.log_figure(fig_conf_matrix,"metrics/confusion_matrix.png")