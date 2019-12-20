import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import random

# enviroment booleans
DEBUG = True
PLOTS = True
SEED = 805
#SEED = random.randint(0, 1000)

print("=============== STARTING ===============")
print("Debug:%r Plots:%r Seed:%d \n" % (DEBUG,  PLOTS, SEED))

# read data form file CSV into pandas data structure
data = pd.read_csv('data/bank-additional-full.csv', sep=';')

LE = LabelEncoder()
#Work on the Categorical Values
data['job_code'] = LE.fit_transform(data['job'])
data['marital_code'] = LE.fit_transform(data['marital'])
data['education_code'] = LE.fit_transform(data['education'])
data['housing_code'] = LE.fit_transform(data['housing'])
data['loan_code'] = LE.fit_transform(data['loan'])
data['contact_code'] = LE.fit_transform(data['contact'])
data['poutcome_code'] = LE.fit_transform(data['poutcome'])
data['day_of_week_code'] = LE.fit_transform(data['day_of_week']) #maybe
data['month_code'] = LE.fit_transform(data['month']) #maybe
data['subscribed'] = LE.fit_transform(data['y'])

# Drop redundant columns
# drop pday becasue data seems to be inconsistent (ex. pdays says person has not been contacted but previous contacts
# count is greater than 1)
data=data.drop(['job', 'marital', 'education', 'housing', 'loan', 'contact', 'poutcome', \
                'day_of_week', 'month', 'default', 'pdays', 'y'], axis=1)

if DEBUG:
    export_csv = data.to_csv (r'cleaned_input_data.csv', index=None, header=True)

# Generator feature selectors to use for training
selector = [
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

# divide out the given training data into training (80%) and validation (20%)
X_train, X_test, y_train, y_test = train_test_split(data[selector], data['subscribed'], test_size=0.2, random_state=SEED)

if DEBUG:
    X_train.to_csv(r'X_train.csv', index=None, header=True)
    X_test.to_csv(r'X_test.csv', index=None, header=True)


# scale the data:
# transforms the data to center it by removing the mean value of each feature, then scale it by dividing non-constant
# features by their standard deviation.
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# generate MLP with 1 hidden layer of 9 neurons (18input/2)
mlp = MLPClassifier(hidden_layer_sizes=(8,),
                    max_iter=500,
                    activation = 'relu',
                    solver='adam',
                    verbose=DEBUG,
                    early_stopping=True,
                    validation_fraction=0.2,
                    random_state=SEED)
mlp.fit(X_train_scaled, y_train)

if PLOTS:
    # plot the error curve
    plt.plot(mlp.loss_curve_,  label='Model Error vs Epoch')
    plt.title('Learning Loss Function')
    plt.xlabel('Loss')
    plt.ylabel('Epoch')
    plt.savefig("output/accuracy_vs_epoch.png", bbox_inches='tight', dpi=200, pad_inches=0.5)
    plt.close()

# run the test data to see validity of model
predictions = mlp.predict(X_test_scaled)
cnf_matrix = confusion_matrix(y_test, predictions)

print("***************************************")
print(cnf_matrix)
print("***************************************")

if PLOTS:
    # plot the Confusion matrix
    fig, ax = plt.subplots(1)
    ax = sns.heatmap(cnf_matrix, ax=ax, cmap=plt.cm.Greens, annot=True)
    plt.title('Confusion matrix of random forest predictions')
    plt.ylabel('True category')
    plt.xlabel('Predicted category')
    plt.savefig("output/Confusion_Matrix.png", bbox_inches='tight', dpi=200, pad_inches=0.5)
    plt.close()

print("Training set score: %f" % mlp.score(X_train_scaled, y_train))
print("Test set score: %f" % mlp.score(X_test_scaled, y_test))

print(mean_squared_error(y_test, predictions))