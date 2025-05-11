import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess the dataset
df = pd.read_csv('train.csv')

# Initialize label encoders for categorical features
label_encoders = {}
for column in ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split features and target
X = df.drop(['Reached.on.Time_Y.N'], axis=1)
y = df['Reached.on.Time_Y.N']

# Train the model (RandomForest in this case)
model = RandomForestClassifier()
model.fit(X, y)

# Save the model and label encoders
joblib.dump(model, 'app/model.pkl')
joblib.dump(label_encoders, 'app/label_encoders.pkl')
