from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

# Oversampling 
def oversample_data(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y_encoded)
    return X_resampled, y_resampled, le

# Split 
def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

# Train & Evaluate Model 
def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

#  Run All Classical Models 
def run_classical_models(X, y, skip_nb=False):
    X_res, y_res, le = oversample_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X_res, y_res)

    if not skip_nb:
        train_and_evaluate(MultinomialNB(), X_train, X_test, y_train, y_test, "Naive Bayes")
    else:
        train_and_evaluate(GaussianNB(), X_train, X_test, y_train, y_test, "GaussianNB")

    train_and_evaluate(LinearSVC(C=1.0, loss='squared_hinge'), X_train, X_test, y_train, y_test, "SVM")
    train_and_evaluate(DecisionTreeClassifier(max_depth=10, criterion='entropy'), X_train, X_test, y_train, y_test, "Decision Tree")
    train_and_evaluate(RandomForestClassifier(max_depth=20, n_estimators=100), X_train, X_test, y_train, y_test, "Random Forest")
    train_and_evaluate(AdaBoostClassifier(learning_rate=1.0, n_estimators=100), X_train, X_test, y_train, y_test, "AdaBoost")
