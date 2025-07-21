import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from flask import Flask, request, render_template_string

# Load and preprocess the dataset
df = pd.read_csv('kidney_disease.csv')

df[['pcv', 'wc', 'rc', 'dm', 'cad', 'classification']] = df[['pcv', 'wc', 'rc', 'dm', 'cad', 'classification']].replace(
    to_replace={'\t8400':'8400', '\t6200':'6200', '\t43':'43', '\t?':np.nan, '\tyes':'yes', '\tno':'no', 'ckd\t':'ckd'}
)

df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

df[['pcv', 'wc', 'rc']] = df[['pcv', 'wc', 'rc']].astype('float64')

df.drop(['id', 'sg', 'pcv', 'pot'], axis=1, inplace=True)

categorical_columns = ['rbc', 'pcc', 'pc', 'ba', 'htn', 'dm', 'cad', 'pe', 'ane']
encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

df[['appet', 'classification']] = df[['appet', 'classification']].replace(to_replace={'good':'1', 'ckd':'1', 'notckd':'0', 'poor':'0'})
df[['classification', 'appet']] = df[['classification', 'appet']].astype('int64')

X = df.drop('classification', axis=1)
y = df['classification']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(x_train, y_train)

joblib.dump(rf, 'model_rf.joblib')

# Example of loading the model using joblib
model_rf = joblib.load('model_rf.joblib')

# Flask application
app = Flask(__name__)

category_map = {
    'rbc': {'Normal': 1, 'Abnormal': 0},
    'pcc': {'Present': 1, 'Not Present': 0},
    'pc': {'Normal': 1, 'Abnormal': 0},
    'ba': {'Present': 1, 'Not Present': 0},
    'htn': {'Yes': 1, 'No': 0},
    'dm': {'Yes': 1, 'No': 0},
    'cad': {'Yes': 1, 'No': 0},
    'appet': {'Good': 1, 'Poor': 0},
    'pe': {'Yes': 1, 'No': 0},
    'ane': {'Yes': 1, 'No': 0}
}

index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Kidney Disease Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; background: #092756; color: white; text-align: center; padding: 50px; }
        h2 { margin-bottom: 30px; }
        input { margin: 5px; padding: 10px; width: 150px; }
        button { padding: 10px 20px; background: purple; color: white; border: none; cursor: pointer; }
        button:hover { background: darkviolet; }
    </style>
</head>
<body>
    <h2>Kidney Disease Prediction</h2>
    <form method="post" action="{{ url_for('predict') }}">
        <input type="text" name="age" placeholder="age" required>
        <input type="text" name="bp" placeholder="bp" required>
        <input type="text" name="al" placeholder="al" required>
        <input type="text" name="su" placeholder="su" required>
        <select name="rbc" required>
            <option value="Normal">Normal</option>
            <option value="Abnormal">Abnormal</option>
        </select>
        <select name="pc" required>
            <option value="Normal">Normal</option>
            <option value="Abnormal">Abnormal</option>
        </select>
        <select name="pcc" required>
            <option value="Present">Present</option>
            <option value="Not Present">Not Present</option>
        </select>
        <select name="ba" required>
            <option value="Present">Present</option>
            <option value="Not Present">Not Present</option>
        </select>
        <input type="text" name="bgr" placeholder="bgr" required>
        <input type="text" name="bu" placeholder="bu" required>
        <input type="text" name="sc" placeholder="sc" required>
        <input type="text" name="sod" placeholder="sod" required>
        <input type="text" name="hemo" placeholder="hemo" required>
        <input type="text" name="wc" placeholder="wc" required>
        <input type="text" name="rc" placeholder="rc" required>
        <select name="htn" required>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
        </select>
        <select name="dm" required>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
        </select>
        <select name="cad" required>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
        </select>
        <select name="appet" required>
            <option value="Good">Good</option>
            <option value="Poor">Poor</option>
        </select>
        <select name="pe" required>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
        </select>
        <select name="ane" required>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
        </select>
        <br>
        <button type="submit">Predict</button>
    </form>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(index_html)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            int(request.form['age']),
            int(request.form['bp']),
            int(request.form['al']),
            int(request.form['su']),
            category_map['rbc'][request.form['rbc']],
            category_map['pc'][request.form['pc']],
            category_map['pcc'][request.form['pcc']],
            category_map['ba'][request.form['ba']],
            int(request.form['bgr']),
            int(request.form['bu']),
            float(request.form['sc']),
            int(request.form['sod']),
            float(request.form['hemo']),
            int(request.form['wc']),
            float(request.form['rc']),
            category_map['htn'][request.form['htn']],
            category_map['dm'][request.form['dm']],
            category_map['cad'][request.form['cad']],
            category_map['appet'][request.form['appet']],
            category_map['pe'][request.form['pe']],
            category_map['ane'][request.form['ane']]
        ]
        
        # Convert features list to numpy array and reshape for scaler
        features_np = np.asarray(features, dtype=float).reshape(1, -1)
        
        # Scale the features
        scaled_features = scaler.transform(features_np)
        
        # Predict using the model
        prediction = rf.predict(scaled_features)
        
        # Determine output message based on prediction
        if prediction[0] == 1:
            output = 'Kidney Disease Detected'
        else:
            output = 'No Kidney Disease Detected'

        return render_template_string("<h3>Prediction Result: {}</h3>".format(output))

    except Exception as e:
        return render_template_string("<h3>Error encountered: {}</h3>".format(str(e)))

if __name__ == "__main__":
    app.run(debug=True)
