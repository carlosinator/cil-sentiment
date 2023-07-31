# CIL Sentiment Analysis "MINION vs BERT: Efficient Fine-Tuning"

All of our experiments were run using Google Colab. To reproduce the results, using Colab is recommended. Furthermore, all the data and models are stored using the Google Cloud Storage (GCS) system. In GCS we set up a bucket called "cil_2023" containing a floder "final_models".

Short summary for reproducibility: To reproduce the results, run the following notebooks in the given order.
1) Preprocessing.ipynb
2) FastTextModel.ipynb
3) main.ipynb

## Final Report
The report including the plagiarism form is located in "final_report.pdf"

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

In the last chapter of the notebook main.ipynb we compute the Kaggle submission.
