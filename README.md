# Customers' feedback Sentiment 

### classification problem: using the reviews of customers(cx) predict the score given by cx from 1:5.
preprocessing:
              Cleaning the data and balancing the classes, preprocessing the data using stemming and then vectorizing the data.
model:
              Different models were tried and the model with the best accuarcy on the training set is an ensemble of SVC, Logistic regrission and XGBClassifier.

results:
             The trainig set accuracy reached 81%. however, the test set accuracy is 56%. which means that the model is overfitting the training set.

next steps:
             * Using different methods to solve the overfitting problem.
             * Using NN to build a deeplearning model to have better results
          
        
        
