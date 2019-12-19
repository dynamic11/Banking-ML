import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix


DEBUG = True

# read data form file CSV into pandas data structure
data = pd.read_csv('data/bank-additional-full.csv', sep=';')

# This attribute highly affects the output target (e.g., if duration=0 then y=no). Yet, the duration is not known
# before a call is performed. Also, after the end of the call, y is obviously known. Thus, this input should only be
# included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

# drop pday becasue data seems to be inconsistent (ex. pdays says person has not been contacted but previous contacts
# count is greater than 1)

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
# Drop columns

data=data.drop(['job', 'marital', 'education', 'housing', 'loan', 'contact', 'poutcome', \
                'day_of_week', 'month', 'default', 'y'], axis=1)

if DEBUG:
    export_csv = data.to_csv (r'cleaned_input_data.csv', index = None, header=True)


# break up intput and output data
y = data['subscribed']
X = data.drop('subscribed', axis=1)

selector = [
    "age",
    "pdays",
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


# divide out the given training data into training (80%) and validation (20%)
X_train, X_validate, y_train, y_validate = train_test_split(data[selector], data['subscribed'], test_size=0.2, random_state=42)

if DEBUG:
    X_train.to_csv(r'X_train.csv', index=None, header=True)


# normalize the data
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_validate_scaled  = min_max_scaler.fit_transform(X_validate)


mlp = MLPClassifier(hidden_layer_sizes=(17,10,1),max_iter=500, activation = 'relu',solver='adam',  verbose=DEBUG, random_state=1)
mlp.fit(X_train_scaled, y_train)


plt.plot(mlp.loss_curve_,  label='Model Error vs Epoch')
plt.title('Learning Loss Function')
plt.xlabel('Loss')
plt.ylabel('Epoch')
plt.savefig("output/accuracy_vs_epoch.png", bbox_inches='tight', dpi=200, pad_inches=0.5)
plt.close()

predictions = mlp.predict(X_validate_scaled)
cnf_matrix = confusion_matrix(y_validate,predictions)

print(cnf_matrix)

fig, ax = plt.subplots(1)
ax = sns.heatmap(cnf_matrix, ax=ax, cmap=plt.cm.Greens, annot=True)
plt.title('Confusion matrix of random forest predictions')
plt.ylabel('True category')
plt.xlabel('Predicted category')
plt.savefig("output/Confusion_Matrix.png", bbox_inches='tight', dpi=200, pad_inches=0.5)
plt.close()


score = accuracy_score(y_validate,predictions)
print(score)