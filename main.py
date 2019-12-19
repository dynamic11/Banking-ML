import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/bank-additional-full.csv',sep=';')
df.head()


LE = LabelEncoder()
#Work on the Categorical Values
df['job_code'] = LE.fit_transform(df['job'])
df['marital_code'] = LE.fit_transform(df['marital'])
df['education_code'] = LE.fit_transform(df['education'])
df['housing_code'] = LE.fit_transform(df['housing'])
df['loan_code'] = LE.fit_transform(df['loan'])
df['contact_code'] = LE.fit_transform(df['contact'])
df['poutcome_code'] = LE.fit_transform(df['poutcome'])
df['day_of_week_code'] = LE.fit_transform(df['day_of_week']) #maybe
df['month_code'] = LE.fit_transform(df['month']) #maybe
df['subscribed'] = LE.fit_transform(df['y'])
# Drop columns
df=df.drop(['job','marital','education','housing','loan','contact','poutcome','day_of_week','month','y'] ,axis=1)

y = df['subscribed']
X = df.drop('subscribed',axis=1)

# Set seed for reproducibility
np.random.seed(42)
# divide out the given training data into training (80%) and validation (20%)
X_train, X_validate, y_train, y_validate = train_test_split(train[selector], train['Difference'], test_size=0.2, random_state=42)