# MLP: Bank Marketing Data Set
This is a small project that uses MLP to model the relationship between data from a direct marketing campaign
of a Portuguese bank to sales.

*TLDR:* Given some data about a customer we want to know whether the current campaign will be successful for each user.

## Tools Used
* Language: Python
* Sklearn (Scikit-Learn)
* Seaborn
* Pandas

## The data

The data was obtained form the UCI MachineLearning Repository:

[UCI Rep: Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
 
 A description of the data can be found here:
 
 [Description of Data](https://www2.1010data.com/documentationcenter/beta/Tutorials/MachineLearningExamples/BankMarketingDataSet.html)
 
 
 ### Data formatting
 
 Most of the data in the data set is of type strinf (ex. yes, no, unknown) so we used Sklean's Label Encoder to convert
 to integers (0,1,2). The variable named in the format *_code have all been formatted to integer type.
 
 **The 17 input features used:**
 
 ```python
Input_Features = [
    "age",
    "campaign",
    "previous",
    "duration",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "job_code",
    "marital_code",
    "education_code",
    "housing_code",
    "loan_code",
    "contact_code",
    "poutcome_code",
    "day_of_week_code",
    "month_code",
]
```

 **The expected output:**
 
The subscribe parameter indicates whether the client has subscribed for a term deposit
 ```python
model_output = [
    "subscribed"
]
```

### Data splitting
Total number of samples in data: 41,188
* Number of Training data: 26,390 (~64%)
* Number of Validation data: 6,590 (~16%)
* Number of Test data: 8,237 (20%)

## The MLP Model
We are using a 3 layer MLP model using:
* 18 neurons in input layer
* 9 neurons in hidden layer (Sigmoid activation function)
* 1 neuron in output layer

We use the adam solver 

## The Code

### How To Run
To run this code run `main_encoding.py`.

There are two parameters that you can use to *generated output plots* and *debug console logs*.

There is also SEED parameter to define the seeds for the sudo random generators in the code. You can set it as a 
constant to ensure consistency between runs.
 ```python
DEBUG = True
PLOTS = True
SEED = 466
```
## Bonus #1: experiment with dummy variables

In the project you will find a file called `main_encoding_ovrsampling.py`.

The data set provided to us contains more negative outcomes (0) than positive outcomes (1) as seen in the following figure.
This could result in a bias in our model because it has more sample of one kind during training. Ideally, we should have 
more results with positive outcome but for the purposes of this project we can use oversampling to generate additional sample 
with a positive (1) outcome.

<img src="/output/wth_encoding_ovrsampling/Count_Outcomes_orig.png" width="400px">

The method used for oversampling is SMOTE: Synthetic Minority Over-sampling Technique. this method generates synthetic 
data points by taking the vector between k neighbors, and the current data point. It multiplies this vector by a random 
number x (between 1 and 0). It then adds this to the current data point to create the new, synthetic data point.

This change results in a significant improvement in accuracy (~96%).

## Bonus #2: experiment with dummy variables

In the project you will find a file called `main_dummies.py`.

This file runs a similar experiment as the `main.py` file but instead of numerical encoding categorical data it generates
dummy variables.

This generates a total of 62 input features vs. 18 used in the original experiment.




## Good references
* <http://www.columbia.edu/~jc4133/ADA-Project.pdf>
* <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html>
* <https://archive.ics.uci.edu/ml/datasets.php?format=&task=&att=&area=&numAtt=&numIns=greater1000&type=&sort=nameUp&view=table>
* <https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/>
