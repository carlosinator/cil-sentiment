# CIL Sentiment Analysis "Minions: Accurate and Effective"

All of our experiments were run using Google Colab. To reproduce the results, using Colab is recommended. Furthermore, all the data and models are stored using the Google Cloud Storage system.

Short summary for reproducability: To reproduce the results, run the following notebooks in the given order.
1) Preprocessing.ipynb
2) FastTextModel.ipynb
3) main.ipynb

## Preprocessing

We preprocess our data by splitting up the hashtags, e.g. "#ilikethis" to "i like this", and by deleting duplicate tweets. Run Preprocessing.ipynb for that.

## FastText

For the results of FastText, run the notebook FastTextModel.ipynb.

## Training, Loading & Evaluation (all models except FastText)

Defining the setting, training, loading previously saved models and evaluation happens all in the notebook main.ipynb. Concretely, we can do the following.

- Defining the experiment setting, i.e. which models to train and with which parameters
- Training the models
- Loading previously trained and saved models
- Evaluating models
- Loading previously saved evaluation data
- Summarizing all the evaluation results in a table
- Plotting calibration curves

## Prediction for Kaggle submission

The notebook infer_submission.ipynb was used to compute the predictions submitted to Kaggle.
