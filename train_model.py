# Import libraries
import itertools

import matplotlib
matplotlib.use("Agg")  # headless backend: works on GitHub Actions runners
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

sns.set_theme(style="white")

# ---------------------------------------------------------------- Load data
dataset = pd.read_csv("iris.csv")

# "sepal length (cm)" -> "sepal_length"
dataset.columns = [
    c.replace("(cm)", "").strip().replace(" ", "_") for c in dataset.columns
]

# ------------------------------------------------------ Feature engineering
dataset["sepal_length_width_ratio"] = dataset["sepal_length"] / dataset["sepal_width"]
dataset["petal_length_width_ratio"] = dataset["petal_length"] / dataset["petal_width"]

dataset = dataset[[
    "sepal_length", "sepal_width", "petal_length", "petal_width",
    "sepal_length_width_ratio", "petal_length_width_ratio", "target",
]]

# --------------------------------------------------------------- Split data
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=44)

X_train = train_data.drop("target", axis=1).values.astype("float32")
y_train = train_data["target"].values.astype("int32")
X_test = test_data.drop("target", axis=1).values.astype("float32")
y_test = test_data["target"].values.astype("int32")

# ------------------------------------------------------ Logistic Regression
# (multi_class removed: lbfgs already uses multinomial loss)
logreg = LogisticRegression(C=0.0001, solver="lbfgs", max_iter=100)
logreg.fit(X_train, y_train)
predictions_lr = logreg.predict(X_test)

cm_lr = confusion_matrix(y_test, predictions_lr)
f1_lr = f1_score(y_test, predictions_lr, average="micro") * 100
prec_lr = precision_score(y_test, predictions_lr, average="micro") * 100
recall_lr = recall_score(y_test, predictions_lr, average="micro") * 100
train_acc_lr = logreg.score(X_train, y_train) * 100
test_acc_lr = logreg.score(X_test, y_test) * 100

# ------------------------------------------------------------ Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
predictions_rf = rf.predict(X_test)

f1_rf = f1_score(y_test, predictions_rf, average="micro") * 100
prec_rf = precision_score(y_test, predictions_rf, average="micro") * 100
recall_rf = recall_score(y_test, predictions_rf, average="micro") * 100
train_acc_rf = rf.score(X_train, y_train) * 100
test_acc_rf = rf.score(X_test, y_test) * 100


# ------------------------------------------------ Confusion matrix plotting
def plot_cm(cm, target_name, title="Confusion Matrix", normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.title(title)
    plt.colorbar()
    if target_name is not None:
        tick_marks = np.arange(len(target_name))
        plt.xticks(tick_marks, target_name, rotation=45)
        plt.yticks(tick_marks, target_name)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = "{:0.4f}".format(cm[i, j]) if normalize else "{:,}".format(cm[i, j])
        plt.text(j, i, text, horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass))
    plt.savefig("ConfusionMatrix.png", dpi=120)
    plt.close()


target_name = np.array(["setosa", "versicolor", "virginica"])
plot_cm(cm_lr, target_name, title="Confusion Matrix (Logistic Regression)")

# ------------------------------------------------------- Feature importance
feature_df = pd.DataFrame({
    "feature": dataset.columns[:-1],
    "importance": rf.feature_importances_,
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel("Importance", fontsize=14)
ax.set_ylabel("Feature", fontsize=14)
ax.set_title("Random Forest Feature Importances", fontsize=12)
plt.tight_layout()
plt.savefig("FeatureImportance.png", dpi=120)  # save BEFORE closing
plt.close()

# ----------------------------------------------------------- Save the scores
with open("scores.txt", "w") as score:
    score.write("Random Forest Train Accuracy: %2.1f%%\n" % train_acc_rf)
    score.write("Random Forest Test Accuracy: %2.1f%%\n" % test_acc_rf)
    score.write("F1 Score: %2.1f%%\n" % f1_rf)
    score.write("Recall Score: %2.1f%%\n" % recall_rf)
    score.write("Precision Score: %2.1f%%\n" % prec_rf)
    score.write("\n\n")
    score.write("Logistic Regression Train Accuracy: %2.1f%%\n" % train_acc_lr)
    score.write("Logistic Regression Test Accuracy: %2.1f%%\n" % test_acc_lr)
    score.write("F1 Score: %2.1f%%\n" % f1_lr)
    score.write("Recall Score: %2.1f%%\n" % recall_lr)
    score.write("Precision Score: %2.1f%%\n" % prec_lr)

print(open("scores.txt").read())
