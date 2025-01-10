import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Caricare il dataset Iris
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Dividere i dati in input (X) e target (y)
X = data[iris.feature_names]
y = data['target']


# Divisione in training e test set (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scaling dei dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def evaluate_model(model, X_test, Y_test):
    # Effettuare previsioni sui dati di test
    y_pred = model.predict(X_test)
    # Calcolare l'accuracy
    print(f"Accuracy: {accuracy_score(Y_test, y_pred)}")
    # Generare un report dettagliato
    print("\nClassification Report:")
    print(classification_report(Y_test, y_pred))
    # Matrice di confusione
    print("\nConfusion Matrix:")
    print(confusion_matrix(Y_test, y_pred))

# Modello Decision Tree
# 1. Inizializzare il modello Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)

# 2. Addestrare il modello sui dati di training
dt_model.fit(X_train, y_train)

# 3. Valuta il modello
evaluate_model(dt_model, X_test, y_test)


# Risultati Decision Tree:
# - Il modello Decision Tree ha raggiunto un'accuracy perfetta (100%).
# - Precision, recall e F1-score sono pari a 1.0 per tutte le classi.
# - La matrice di confusione non presenta errori.
# Osservazioni Decision Tree:
# - Il modello ha classificato correttamente tutte le istanze nel dataset di test.
# - Questa performance potrebbe indicare overfitting, dato che il dataset è semplice e ben separato.



# Modello Logistic Regression
# 1. Inizializzare il modello Logistic Regression
# Utilizziamo max_iter per assicurarci che il modello converga
lr_model = LogisticRegression(max_iter=200, random_state=42)

# 2. Addestrare il modello sui dati di training
lr_model.fit(X_train, y_train)

# 3. Valuta il modello
evaluate_model(lr_model, X_test, y_test)

# Risultati Logistic Regression:
# - Il modello Logistic Regression ha raggiunto un'accuracy perfetta (100%).
# - Precision, recall e F1-score sono pari a 1.0 per tutte le classi.
# - La matrice di confusione non presenta errori.
# Osservazioni Logistic Regression:
# - I risultati perfetti indicano che il dataset Iris è semplice da separare.
# - Possibile overfitting dovuto alla semplicità e separabilità del dataset.


# Modello KNN
# 1. Inizializzare il modello KNN
# Impostiamo il numero di vicini (k) a 3
knn_model = KNeighborsClassifier(n_neighbors=3)

# 2. Addestrare il modello sui dati di training
knn_model.fit(X_train, y_train)

# 3. Valuta il modello
evaluate_model(knn_model, X_test, y_test)

# Risultati KNN:
# - Il modello KNN ha raggiunto un'accuracy perfetta (100%).
# - Precision, recall e F1-score sono pari a 1.0 per tutte le classi.
# - La matrice di confusione non presenta errori: tutte le istanze sono state classificate correttamente.
# Osservazioni KNN:
# - Poiché il dataset Iris è semplice, KNN si è dimostrato efficace anche con un numero ridotto di vicini (k=3)


# Modello Support Vector Machine
# 1. Inizializzare il modello SVM
# Utilizziamo un kernel lineare
svm_model = SVC(kernel='linear', random_state=42)

# 2. Addestrare il modello sui dati di training
svm_model.fit(X_train, y_train)

# 3. Valuta il modello
evaluate_model(svm_model, X_test, y_test)


# Risultati Support Vector Machine:
# - Il modello SVM ha ottenuto un'accuracy di 96.67%
# - Precision, recall e F1-score sono elevati per tutte le classi, con una leggera diminuzione per la classe `1` (recall = 0.89).
# - La matrice di confusione mostra un errore: un'istanza della classe `1` è stata classificata come classe `2`.
# Osservazioni Support Vector Machine:
# - SVM si è dimostrato robusto anche con un kernel lineare, ma potrebbe beneficiare di un kernel più sofisticato per migliorare ulteriormente le performance.
