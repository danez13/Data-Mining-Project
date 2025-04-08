# Data Mining Project #2
### Total points: 100
## Objective
The US food supply is dominated by ultra-processed foods. The purpose of this project is to implement
**binary classifiers to identify food items that are not ultra-processed** _(positive class, 1)_. It is recommended
to use Python to implement the algorithms. If you wish to use another language, your team needs to
agree, and you also need to let the instructor know.


Teams: You will be in the same team as in project #1.


## Dataset
The dataset includes a number of food items, and some features that describe them. At this point, you
will be provided with one file which you can use however you see fit to build your classifiers. At a later
point, we will also share a test file, that has identical structure as the first file.
The columns/features are the following:
1. original_ID

2. name

3. store

4. food category

5. brand

6. **f_FPro_class** -> class labels
   * takes values [0,1,2,3], where 0 corresponds to raw products, 3
corresponds to ultra-processed foods.
7. price

8. price percal

9. package_weight

10.  Protein

11.  Total Fat

12.  Carbohydrate

13.  Sugars, total

14.  Fiber, total dietary

15.  Calcium

16. Iron

17. Sodium

18. Cholesterol

19. Fatty acids, total saturated

Note that columns 1, 2, and 5 are almost unique per item.

## Data Preprocessing
As a first step, you need to familiarize yourselves with the data. Note that you will need to also transform
the class labels from [0,1,2,3] to binary _(f_FPro_class=3 -> label=0, else label=1)_. Also, explore each
feature, and compute their statistics, distribution of values, etc. Feel free to transform the data, remove
features, or create new ones.

## Classification models
1. First, you need to come up with a simple classifier that would serve as the baseline for your
comparison. You can decide what example this will be. The important thing to remember here is
that this **baseline model** will not actually “learn” a model. You can base your decisions on
average, popularity, etc.

2. Build a **Decision Tree** model; examine <ins>at least</ins> the hyperparameter `min_samples_leaf`.

3. Build a **Random Forest** model; examine <ins>at least</ins> the hyperparameters `max_features` and
`n_estimators`.

## Model Selection & Evaluation
Split the data into two parts: training and test sets. Set the random seed to your TeamID, in order to
replicate your results as needed. For each hyperparameter, try at least 4 values.

For the DT and RF models:
* For model selection, create a table _(and store it in a file, e.g., `t09_DTall.csv`, where 09
corresponds to the team ID)_ with the columns describing the different hyperparameters tried and
the accuracy and F1-Score achieved on the training/validation/test sets for these specific
hyperparameters.
  * Select the model with the best validation accuracy.
* For model evaluation, create a file _(e.g., `t09_Tbest.csv`)_ with one line where you will report
the accuracy, recall, precision, and F1-Score on the validation and test set of the best model.

## Evaluation metrics
You will evaluate the models using Accuracy, Precision, Recall, and F1 score. You can use the built-in
functions.

## Plots
You also need to create a code to read the output files from the other models, and generate the following
plots:
* **Plot needed #1**: For the DT model, create a plot to visualize the accuracy of the training/validation/test
sets, where the x-axis is the `min_samples_leaf` and the y-axis is the accuracy achieved.

* **Plot needed #2:** For the FT model, plot in one graph the best test accuracy for the different number of
base classifiers used, where the x-axis is the `n_estimators` and the y-axis is the test accuracy achieved.
There will be two lines in this graph corresponding to the minimum and maximum value that you tried for
`max_features`.

## Notes
* All filenames submitted or generated by your code should be starting with “`t09`”, where the two
digits correspond to the team ID.

* Your code should have clear comments to help us understand your logic and flow.

* You can use the scikit-learn package to build your models.

* Please, list all the sources that you will use. You can even use LLMs, but only if you properly
mention it. Also, keep in mind, that in the end of the day, each member of the team will be
responsible for their code.

* When you are comparing models, the metrics should be computed over the same data, i.e.,
train/validation/test data should be the same across models. So, remember to properly use seeds
_(and the “`random_state`” on the models)_ and verify that the data splits are the same. That can
also help your code to be reproducible.

* Your code should be reproducible.

* If you are not following the guidelines, we might have to subtract points from your submission.

## What to submit
Your report needs to have the following elements:
• Code files:
  1. Data analysis and preprocessing
  
  2. Baseline model
  
  3. DT model

  4. FR model 
  
  5. Plotting

• The csv files generated by your codes.

• Report:
  1. How did your team split the work?
  
  2. How did your team communicate and work together? You need to keep track of the dates,
times, who was present, and whether you met in person or online for all your interactions
regarding the project.
  
  3. References and explanation of their use.
  
  4. For each feature, describe its characteristics, _e.g., when applicable: max and min values, range
of values, number of different attribute values, distributions, etc._
  
  5. Explain all the decisions that you made during data preprocessing.
  
  6. Explain how you set up your experiments for data selection and parameter tuning.
  
  7. Focusing on the DT model and the “`t09_DTall.csv`” file, compare the test performance
if instead of the validation accuracy, we would use the training accuracy _(an optimistic
estimate of the performance)_ for selection. What do you observe?
  
  8. Compare the best-performing models _(and create a table)_ from the “`t09_*best.csv`”
files. Which model performs the best? How is the best performance compared to the baseline
performance? Are different metrics telling us a different story?
  
  9. Plots #1 and #2 generated in the last step. What can you notice in each graph?
