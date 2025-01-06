# Iris Classification Project

## 1. Titolo del Progetto
Iris Classification Project: An Introduction to Machine Learning

## 2. Breve Descrizione
Questo progetto mira a costruire un modello di Machine Learning per classificare i fiori Iris in tre specie diverse (**Setosa**, **Versicolor**, **Virginica**) basandosi su caratteristiche misurate come la lunghezza e la larghezza dei petali e sepali.

L'obiettivo principale è utilizzare i concetti fondamentali del Machine Learning applicando algoritmi come **Decision Tree**, **Logistic Regression**, **K-Nearest Neighbors (KNN)**, e **Support Vector Machines (SVM)** su un dataset standard.

## 3. Dataset Utilizzato
Il dataset utilizzato è **Iris Dataset**, disponibile nella libreria `scikit-learn`.

- **Dimensioni**: 150 righe, 5 colonne
- **Caratteristiche**:
  - Lunghezza del sepalo (cm)
  - Larghezza del sepalo (cm)
  - Lunghezza del petalo (cm)
  - Larghezza del petalo (cm)
- **Etichetta (target)**: Specie di Iris (Setosa, Versicolor, Virginica)

## 4. Metodologia
1. **Caricamento dei dati**: Utilizzo del dataset Iris da `scikit-learn`.
2. **Preprocessamento**: Divisione in set di training e test.
3. **Addestramento**: Implementazione e confronto di diversi algoritmi di classificazione:
   - Decision Tree
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machines (SVM)
4. **Valutazione**: Confronto delle metriche di accuratezza per selezionare il modello migliore.

## 5. Requisiti (Dipendenze)
Assicurati di avere Python installato (versione 3.7+).

Dipendenze richieste:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

Puoi installare tutte le dipendenze con:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 6. Installazione e Utilizzo
1. Clona il repository del progetto:
   ```bash
   git clone <https://github.com/davidequattrocchi10/iris_classification.git> 
   ```
2. Crea e attiva un ambiente virtuale:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Su Windows: venv\Scripts\activate
   ```
3. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```
4. Esegui lo script principale:
   ```bash
   python main.py
   ```
   
## 7. Autore e Licenza
**Autore**: Davide Quattrocchi

**Licenza**: Questo progetto è rilasciato sotto la licenza MIT.

