import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Loan_predication.csv")

df.drop("Loan_ID", axis=1, inplace=True)

# Handle missing values
df.fillna({
    "LoanAmount": df["LoanAmount"].median(),
    "Loan_Amount_Term": df["Loan_Amount_Term"].median(),
    "Credit_History": df["Credit_History"].mode()[0]
}, inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in ["Gender","Married","Dependents","Education",
            "Self_Employed","Property_Area","Loan_Status"]:
    df[col] = le.fit_transform(df[col])

# Feature Engineering (KEY IMPROVEMENT)
df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["Loan_Income_Ratio"] = df["LoanAmount"] / (df["Total_Income"] + 1)

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3
)

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model Accuracy:", round(accuracy*100, 2), "%")

pickle.dump(model, open("loan_model.pkl", "wb"))
