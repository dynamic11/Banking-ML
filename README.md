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
* 17 neurons in input layer
* 8 neurons in hidden layer (ReLU activation function)
* 1 neuron in output layer

We use the adam solver 

## The Code

### How To Run
To run this code run main.py

There are to parameters that you can use to *generated output plots* and *debug console logs*

 ```python
DEBUG = True
PLOTS = True
```

## Good references
* <http://www.columbia.edu/~jc4133/ADA-Project.pdf>
* <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html>
* <https://archive.ics.uci.edu/ml/datasets.php?format=&task=&att=&area=&numAtt=&numIns=greater1000&type=&sort=nameUp&view=table>
* <https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/>