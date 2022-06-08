# NLP_Sentiment - UNIBA_NLP2122_Leshi

NLP project for the exam - Leshi group

Kaggle challenge: [Sentiment Analysis on Movie Reviews](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview)

# Installation

Simply run `pip install -r requirements.txt`

**IMPORTANT**: on jupyter notebooks please restart the kernel after the installation

# General Information

The source code is organized into 3 different packages (*+1 for utils*), one for each approach described in the
[presentation](UNIBA_NLP2122_Leshi.pdf)
* Each file contains a `main` method which guides you in performing the various experiments

# Computing best submission file

The CSV submitted that returned the best score is in the root of the repository ([here](bert-base-uncased_with_pos_split_2.csv))

If you want to regenerate it, run the [`obtain_best_solution.py`](obtain_best_solution.py) module
* Note that the code in this module will download the already fine-tuned model.
* If you wish to start from scratch, visit the [following module](src/transformers/transformers_approach.py) and replace
appropriately the variables using Bert model with POS strategy
  * In this case the process is very expensive and it may take a lot of time
  * Since dataset is shuffled with a random seed and the hyperparameters search is also randomly initialized, 
the result may slightly differ

# Hyperparameters trials

Inside the [hyperparameters_search_output folder](hyperparameters_search_output) you can find the final report containing
the 8 trials done by ray tune when searching for best hyperparameters
