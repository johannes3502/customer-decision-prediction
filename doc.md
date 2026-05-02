
# Project: Customer Decision Prediction – ML Pipeline Documentation

**Date:** 2026-05-02 (Baseline Evaluation Complete & Finalized)  
**Coordinator:** Johannes (Pipeline & Evaluation)  
**Course:** Introduction to Machine Learning  
**Team:** Group 1  

---

## 1. Überblick: Die ML-Pipeline

Diese Dokumentation beschreibt die gesamte **automatisierte Datenpipeline** vom Download über die Vorbereitung bis zur Baseline-Evaluierung. Das System ist so gestaltet, dass jedes Gruppenmitglied einfach alle Schritte reproduzieren kann.

### Die vier Phasen

| Phase | Skript | Ausgang | Zweck |
|-------|--------|---------|-------|
| **1. Download** | `src/utils/download.py` | `data/raw/*.csv` | Automatisches Abrufen der öffentlichen Datensätze |
| **2. Preprocessing** | `src/setup_data.py` | `data/processed/*_clean.csv` | Einheitliche 7-Schritte-Reinigung aller Datensätze |
| **3. Explorative Analyse (EDA)** | `src/analysis.py` | `reports/summary_*.json`, Plots | Verständnis der Daten, Metadaten für Validierung |
| **4. Baseline-Evaluierung** | `src/evaluation.py` | `reports/baseline_*.csv` | Vergleich von 3 Klassifizierern mit CV und Hold-out Test |

Jede Phase ist **reproduzierbar** (alle Random Seeds sind gesetzt) und **unabhängig lauffähig** (mit Caching/Skippping bei bereits vorhandenen Dateien).

---

## 2. Getting Started: So startest du die Pipeline

### 2.1 Schritt 1: Daten herunterladen und vorverarbeiten

```bash
cd src
python setup_data.py
```

**Was passiert:**
- Lädt Telco Churn, Bank Marketing, E-Commerce Shopper von ihren Quellen herunter (einmalig)
- Wendet 7-Schritte-Preprocessing auf alle drei Datensätze an
- Speichert saubere Versionen in `data/processed/`
- Ausgabe: `telco_clean.csv`, `bank_clean.csv`, `ecom_clean.csv`

**Dauer:** ~30–60 Sekunden (abhängig von Internetverbindung beim ersten Lauf)

### 2.2 Schritt 2: Explorative Analyse durchführen

```bash
cd src
python analysis.py
```

**Was passiert:**
- Analysiert die **rohen** (vor Preprocessing) Datensätze
- Schreibt JSON-Summaries mit Metadaten (Shape, Typen, Missing Values, Korrelationen, etc.)
- Generiert Korrelations-Heatmaps
- Speichert alles in `reports/`
- Ausgabe: `summary_telco.json`, `summary_bank.json`, `summary_ecom.json`, Heatmaps

**Dauer:** ~10–20 Sekunden

**Wichtig:** Diese Skripte sind idempotent. Du kannst sie mehrfach ausführen; bereits heruntergeladene Dateien werden übersprungen.

### 2.3 Schritt 3: Baseline-Klassifizierer evaluieren

```bash
cd src
python evaluation.py
```

**Was passiert:**
- Validiert die bereinigten Daten gegen die EDA-Summaries
- Trainiert 3 Baseline-Modelle (Logistic Regression, k-NN, Naive Bayes)
- Verwendet 5-Fold Kreuzvalidierung + Hold-out Test (80/20 Split)
- Berechnet Accuracy, F1-Score, ROC-AUC
- Speichert Ergebnisse in CSVs
- Ausgabe: `baseline_cv_results.csv`, `baseline_test_results.csv`

**Dauer:** ~1 Minuten (abhängig von Hardware)

**Nach diesem Lauf kannst du die Ergebnisse einlesen:**

```python
import pandas as pd
cv_results = pd.read_csv("../reports/baseline_cv_results.csv")
test_results = pd.read_csv("../reports/baseline_test_results.csv")
print(cv_results)
print(test_results)
```

---

## 3. Preprocessing: Die 7-Schritte-Pipeline

Alle Datensätze werden durch die folgende einheitliche Pipeline verarbeitet:

| Schritt | Operation | Grund |
|---------|-----------|-------|
| **1. Duplikate** | Entferne exakte Duplikat-Zeilen | Redundanzen beseitigen |
| **2. Column-Namen** | Normalisiere: `strip()`, ersetze Spaces/Slashes durch `_` | Konsistente Spalten-Referenzen |
| **3. String-Zellen** | Trimme Leading/Trailing Whitespace | Placeholder-Erkennung robuster |
| **4. Placeholders** | Ersetze `"unknown"`, `"?"`, `""` → NaN; `999` nur in bestimmten Spalten | Fehlende Werte explizit kennzeichnen |
| **5. Typ-Konversion** | Konvertiere object-Spalten, die zu >95% numerisch sind | Type-Konsistenz |
| **6. Imputation** | Numerisch: Median; Kategorial: Mode | NaN-Werte füllen |
| **7. Target-Encoding** | Kodiere binäres Target zu `0`/`1` (int) | Modeling-Ready Format |

### Wichtiger Hinweis zu Schritt 6 (Imputation)

Die **globale Imputation** (Schritt 6) ist ein minimaler Kompromiss im Kursrahmen:
- Median und Mode werden über den **gesamten** (nach Duplikaten bereinigten) Datensatz berechnet
- Dies ist nicht ideal für finale Modelle, aber für Baselines akzeptabel
- **Die Evaluierung behebt das:**  In `evaluation.py` wird jede Imputation/Skalierung **nur auf Trainingsdaten** pro CV-Fold durchgeführt → keine Data Leakage zwischen Train und Test

### Beispiele aus der Praxis

- **Telco**: 11 fehlende Werte in `TotalCharges` → mit Median aufgefüllt
- **Bank**: `pdays=999` in 39.673 Einträgen identifiziert und als Placeholder ersetzt → Median
- **Ecom**: `ProductRelated_Duration=999` in 3 Einträgen → NaN → Median

---

## 4. Explorative Analyse (EDA): Was die Daten uns sagen

Nach dem Preprocessing wird die EDA auf den **rohen** Datensätzen durchgeführt. Dies ist gewollt, um reale Datenprobleme zu dokumentieren.

### Artefakte der EDA

#### JSON-Summaries
Für jeden Datensatz wird ein `summary_<dataset>.json` erstellt mit:

- **Shape**: Anzahl Zeilen, Spalten
- **Dtypes**: Datentypen pro Spalte (direkt weiterverwendet in Evaluierung!)
- **Missing**: Fehlende Werte und erkannte Placeholder
- **Target-Verteilung**: Klassenungleichgewichte quantifiziert
- **Numerische Outlier**: IQR-Methode, Counts und Ranges
- **Duplikate**: Exakte Duplikat-Zeilen
- **Korrelationen**: Top-Korrelationen mit Target und starke Paare (|r| > 0.9)

Die JSON-Summaries werden **direkt von `evaluation.py` verwendet**, um:
1. Die bereinigten Daten zu validieren (Column-Namen, Typen)
2. Feature-Typen automatisch zu erkennen (numerisch vs. kategorial)
3. Datenkonsistenz sicherzustellen

#### Text-Report
`eda_report.txt` enthält lesbare Zusammenfassungen aller Analysen – gut für die Dokumentation und Diskussion im Notebook.

#### Korrelations-Heatmaps
`plots/corr_heatmap_<dataset>.png` zeigen visuell, welche Features miteinander korrelieren und welche mit dem Target.

### Wichtige Erkenntnisse pro Datensatz

**Telco Churn (7.043 Zeilen, 20 Features):**
- Klassenverteilung: 73% Nicht-Churn, 27% Churn (moderates Ungleichgewicht)
- Top Korrelationen mit Target: `tenure` (r=–0.35), `MonthlyCharges` (r=0.19)
- 11 fehlende Werte in `TotalCharges`
- 22 exakte Duplikate (0.31%)

**Bank Marketing (41.188 Zeilen, 21 Features):**
- Klassenverteilung: **89% Keine Anmeldung, 11% Anmeldung** (extremes Ungleichgewicht!)
- **999 als Placeholder:** 39.673 Einträge in `pdays` (97% der Spalte!)
- Starke wirtschaftliche Korrelationen: `emp.var.rate` ↔ `euribor3m` (r=0.97), `emp.var.rate` ↔ `nr.employed` (r=0.91)
- Top Korrelationen mit Target: `duration` (r=0.41), `pdays` (r=–0.32), `nr.employed` (r=–0.35)

**E-Commerce (12.330 Zeilen, 18 Features):**
- Klassenverteilung: **85% Kein Kaufabsicht, 15% Kaufabsicht** (starkes Ungleichgewicht)
- Top Korrelationen mit Target: `PageValues` (r=0.49), `ExitRates` (r=–0.21)
- Behavioral Features (Besuchsdauer, Bounce Rates) sind gute Predictoren
- Starke Korrelation: `BounceRates` ↔ `ExitRates` (r=0.91)

---

## 5. Baseline-Evaluierung: Methodik und Design

### 5.1 Die drei Modelle

| Modell | Komplexität | Kernidee | Wann gut? | Schwäche |
|--------|-----------|----------|----------|---------|
| **Logistic Regression** | Niedrig (linear) | Lineare Trennoberfläche mit Wahrscheinlichkeiten | Strukturierte Daten, Interpretierbarkeit | Krummlinige Grenzen |
| **k-NN (k=5)** | Mittel (memory-based) | Klassifiziere nach Mehrheit der k nächsten Nachbarn | Kleine Datenmengen, komplexe Ränder | Hochdimensional, Imbalance, langsam |
| **Naive Bayes** | Niedrig (probabilistisch) | Bayes-Theorem mit Feature-Unabhängigkeits-Annahme | Text-Klassifikation, schnelle Baseline | Unrealistische Unabhängigkeits-Annahme |

### 5.2 Train/Test Split: Unterschied zwischen Datensätzen

**Für Telco und E-Commerce:**
- 80% Training, 20% Test (zufällige Aufteilung, stratifiziert nach Target)
- `random_state=42` für Reproduzierbarkeit
- Standard-Ansatz für unstrukturierte/unkritische Daten

**Für Bank – Zeitblind Split (Temporale Validierung):**
- Der Bank-Datensatz hat zeitliche Anteile: `month` (1–12), `day_of_week` (Mon–Fri)
- **Preprocessing sortiert diese Daten chronologisch**: Monat aufsteigend, dann Wochentag
- **Datensplit:** Erste 80% (zeitlich älteste) → Training; letzte 20% (zeitlich neueste) → Test
- **Grund:** Marketing-Kampagnen werden auf historischen Daten trainiert und auf zukünftige Kundinnen angewendet. Zufällige Vermischung würde zu optimistischen Scores führen (Look-ahead Bias)

### 5.3 Cross-Validation: 5-Fold, aber mit Unterschied

**Telco und E-Commerce:**
- Stratified 5-Fold: `StratifiedKFold` (Klassen-Anteil bleibt in jedem Fold erhalten)

**Bank:**
- `TimeSeriesSplit` mit k=5: **In jedem Fold liegen Trainingsdaten zeitlich VOR Validierungsdaten** – keine Look-ahead Verzerrung
- Dies ist konsistent mit dem Hold-out Test (der auch zeitblind ist)

**Warum das wichtig ist:**
- Bei Bank sind CV-Scores möglicherweise niedriger als bei Telco/Ecom, weil das Modell auf "Vergangenheit" trainiert und auf "Zukunft" testet
- Konzeptdrift (Economic cycles) könnte dazu führen, dass ältere Muster nicht auf neue Daten übertragen
- **Das ist kein Overfitting – das ist Realität!**

### 5.4 Preprocessing-Pipelines (Inside the Loop)

Für jedes Modell wird eine eigene scikit-learn Pipeline definiert:

```
Logistic Regression:    StandardScaler → OneHotEncoder → LogisticRegression
k-NN:                   StandardScaler → OneHotEncoder → KNeighborsClassifier
Naive Bayes:            Passthrough     → OneHotEncoder → GaussianNB
```

**Kritisch: Diese Pipelines werden nur auf Trainingsdaten pro CV-Fold gefittet!**

```python
for train_idx, val_idx in cv_splitter:
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Pipeline wird AUF TRAIN gefittet
    pipeline.fit(X_train, y_train)
    # Dann auf VAL angewendet
    y_pred = pipeline.predict(X_val)
```

**Warum?** StandardScaler berechnet Mean/Std von X_train; OneHotEncoder merkt sich Kategorien von X_train. Wenn das auf Val-Daten gemacht würde, wäre das **Data Leakage** – Validierungsdaten würden das Training beeinflussen.

**Feature-Typ-Erkennung:** Numerische vs. kategoriale Features werden aus der EDA-JSON `dtypes` abgelesen (nicht aus `select_dtypes()` im Runtime), um Konsistenz sicherzustellen.

### 5.5 Metriken: Warum drei?

| Metrik | Definition | Bereich | Wann wichtig? |
|--------|-----------|---------|--------------|
| **Accuracy** | (TP + TN) / (TP + FP + FN + TN) | 0–1 (höher besser) | Balanced Classes; aber **verzerrt bei Imbalance** |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) | 0–1 (höher besser) | **Imbalanced Classes** – fair zu Minderheit |
| **ROC-AUC** | Area Under Receiver-Operating-Characteristic Curve | 0–1 (höher besser) | Schwellenwert-unabhängig; robust gegen Imbalance |

**Beispiel – Bank (89% Mehrheitsklasse):**
- Dummy-Modell (immer "no" sagen): Accuracy = 89% ✓ (täuscht!)
- Unser LogReg (CV): Accuracy = 92.71% ✓, F1 = 0.4750, ROC-AUC = 0.9283
- F1 und ROC-AUC zeigen, dass das Modell die Minderheit tatsächlich vorhersagt!

---

## 6. Ergebnisse: Was die Zahlen bedeuten

### 6.1 CV-Ergebnisse (5-Fold, Mean ± Std)

| Datensatz | Modell | Accuracy | F1 | ROC-AUC |
|-----------|--------|----------|-----|---------|
| **Telco** | Logistic Regression | 80.19% ± 1.16% | 0.592 ± 0.030 | 0.846 ± 0.013 |
| | k-NN | 76.50% ± 0.71% | 0.547 ± 0.016 | 0.783 ± 0.007 |
| | Naive Bayes | 69.65% ± 0.91% | 0.597 ± 0.010 | 0.821 ± 0.012 |
| **Bank** | Logistic Regression | 92.71% ± 1.80% | 0.475 ± 0.028 | 0.928 ± 0.018 |
| | k-NN | 92.40% ± 1.83% | 0.417 ± 0.057 | 0.838 ± 0.032 |
| | Naive Bayes | 88.42% ± 2.69% | 0.350 ± 0.079 | 0.819 ± 0.047 |
| **Ecom** | Logistic Regression | 88.20% ± 0.49% | 0.497 ± 0.027 | 0.890 ± 0.014 |
| | k-NN | 87.64% ± 0.55% | 0.496 ± 0.030 | 0.786 ± 0.011 |
| | Naive Bayes | 79.49% ± 0.66% | 0.484 ± 0.018 | 0.812 ± 0.009 |

### 6.2 Test-Ergebnisse (Hold-out 20%)

| Datensatz | Modell | Accuracy | F1 | ROC-AUC |
|-----------|--------|----------|-----|---------|
| **Telco** | Logistic Regression | 80.55% | 0.604 | 0.842 |
| | k-NN | 76.37% | 0.564 | 0.790 |
| | Naive Bayes | 69.55% | 0.593 | 0.808 |
| **Bank** | Logistic Regression | 87.76% | 0.629 | 0.912 |
| | k-NN | 85.13% | 0.498 | 0.846 |
| | Naive Bayes | 81.99% | 0.538 | 0.838 |
| **Ecom** | Logistic Regression | 88.94% | 0.541 | 0.902 |
| | k-NN | 87.96% | 0.530 | 0.811 |
| | Naive Bayes | 80.25% | 0.508 | 0.830 |

### 6.3 Interpretation

**Logistic Regression gewinnt überall:**
- Highest Accuracy auf allen drei Datensätzen
- Usually highest ROC-AUC
- Suggeriert: **Lineare Trennbarkeit dominiert** – die einfachsten Modelle sind oft die besten für diese Daten

**Naive Bayes leidet besonders bei Bank:**
- F1 sinkt von 0.597 (Telco) auf 0.350 (Bank)
- **Grund:** Bank hat extreme Korrelationen zwischen Features (z. B. `emp.var.rate` ↔ `euribor3m`, r=0.97)
- Die Unabhängigkeits-Annahme ist stark verletzt

**k-NN hat Probleme bei Imbalance:**
- Bei Bank: F1 = 0.417 (vs. LogReg: 0.475)
- **Grund:** Mit 89% Mehrheitsklasse finden die 5 nächsten Nachbarn oft nur Mehrheits-Punkte
- Hochdimensionale Räume verschärfen das Problem

### 6.4 CV vs. Test: Overfitting oder Realität?

**Telco:**
- CV Accuracy: 80.19%, Test Accuracy: 80.55% (Δ = +0.36%) → Stabil, keine Überfitting-Anzeichen

**Bank – Besonderheit:**
- CV Accuracy: 92.71%, Test Accuracy: 87.76% (Δ = –4.95%)
- **Das ist NICHT Overfitting!** Das ist Generalisierungslücke durch zeitliche Validierung:
  - CV testet auf zeitlich vermischten Daten
  - Test testet auf "Zukunft" (neueste 20% der Zeitachse)
  - Konzeptdrift und Wirtschaftszyklen spielen eine Rolle
- Das ist **realistisch und gewollt** – echte Marketing-Kampagnen haben das auch!

**Ecom:**
- CV Accuracy: 88.20%, Test Accuracy: 88.94% (Δ = +0.74%) → Stabil

### 6.5 Der Accuracy-Trap bei Imbalance (vor allem Bank!)

Bank zeigt ein klassisches Problem:

| Modell | Accuracy | F1 | ROC-AUC |
|--------|----------|-----|---------|
| Dummy (immer "no") | **89%** 🚨 | 0 | 0.5 |
| Unser LogReg | 87.76% | **0.629** ✓ | **0.912** ✓ |

- Accuracy sagt: Dummy ist besser (89% > 87.76%)
- F1 und ROC-AUC sagen: LogReg ist viel besser
- **Lektion:** Bei Imbalance-Problemen: **F1 und ROC-AUC vertrauen, nicht Accuracy!**

---

## 7. Bekannte Probleme und Kompromisse

### 7.1 Row-Count Mismatch zwischen EDA und Evaluierung

- **EDA** läuft auf rohen Daten mit Duplikaten (z. B. Telco: 7.043 Zeilen)
- **Evaluierung** läuft auf Daten, bei denen Duplikate entfernt wurden (z. B. Telco: 7.021 Zeilen)
- **Lösung:** Der Code fängt das mit einer Warnung (`"Row count mismatch detected; validation continues."`) ab und arbeitet korrekt trotzdem weiter
- **Impact:** Minimal, validiert aber, dass die Daten konsistent bearbeitet werden

### 7.2 Globale Imputation statt Pipeline-Imputation

- **Schritt 6 des Preprocessing:** Median/Mode werden über den **ganzen** Datensatz berechnet
- **Ideale Alternative:** Imputation mit SimpleImputer **nur auf Training-Folds** (wie Scaler)
- **Warum nicht gemacht:** Für Baselines akzeptabel, schneller in der Praxis
- **Später (Task 3/4):** Könntet ihr Pipelines verwenden, die auch Imputation inside the CV-Loop machen

### 7.3 Column-Namen-Normalisierung

- EDA und Evaluierung müssen die Column-Namen konsistent interpretieren
- Der Code normalisiert diese (Spaces → Underscores, strip, etc.)
- Falls Fehler auftreten: Prüfe, ob `evaluate.py` und die JSON-Summaries gleiche Column-Namen haben

### 7.4 Pfad-Verweis zu Reports

- `REPORTS_DIR` ist auf `PROJECT_ROOT / "reports"` konfiguriert
- Falls beim ersten Ausführen von `analysis.py` oder `evaluation.py` Fehler "Verzeichnis nicht gefunden" auftreten: Stelle sicher, dass das `reports/` Verzeichnis existiert oder dass `REPORTS_DIR` korrekt gesetzt ist

### 7.5 .gitignore aktualisiert

- `data/raw/` und `data/processed/` sind ignoriert → die Dateien werden nicht in Git committed
- `reports/baseline_*.csv` sind ignoriert
- **Folge:** Jeder Gruppenmitglied muss die Pipeline selbst starten, um die Daten zu erhalten
- **Das ist gewollt** – so wird sichergestellt, dass alle die gleiche Version verwenden und der Code reproducible ist

---

## 8. Anleitung für die Weiterarbeit

### 8.1 Für das Meilenstein-Notebook (Milestone 1)

Diskutiere im Notebook folgende Punkte:

1. **Klassifizierungsaufgabe und Datensätze:**
   - Warum diese drei Datensätze? Was sind realistische Geschäftskontexte?
   - Klassenungleichgewichte und deren Implikationen

2. **Methodologie:**
   - Warum 80/20 Split, warum Stratification?
   - Warum Temporal Split bei Bank? (Look-ahead Bias, Konzeptdrift)
   - Warum 5-Fold CV?
   - Metriken-Wahl: Wann Accuracy, wann F1, wann ROC-AUC?

3. **Ergebnisse verstehen:**
   - Warum gewinnt Logistic Regression überall?
   - Warum leidet Naive Bayes bei Bank?
   - Was bedeutet die 4.95%-Differenz bei Bank zwischen CV und Test?

4. **Erkenntnisse aus EDA:**
   - Starke Korrelationen bei Bank – was bedeutet das?
   - Placeholder-Werte in `pdays` – wie wurde das gehandhabt?
   - Implikationen der Imbalance für Task 3

5. **Limitationen und nächste Schritte:**
   - Warum sind Baselines wichtig, bevor man zu SVM/MLP/RF/XGB geht?
   - Welche Features treiben die Vorhersagen? (Feature Importance in Task 4)
   - Hyperparameter-Tuning Ansatz?

**Code zum Einbinden der Ergebnisse in das Notebook:**
```python
import pandas as pd
cv_results = pd.read_csv("../src/reports/baseline_cv_results.csv")
test_results = pd.read_csv("../src/reports/baseline_test_results.csv")

# Visualisierung der Ergebnisse
import matplotlib.pyplot as plt
cv_results.groupby('Dataset')[['Accuracy_mean', 'F1_mean', 'ROC_AUC_mean']].plot(kind='bar')
plt.title("Baseline CV Results")
plt.show()
```

### 8.2 Für Task 3 und 4 (Erweiterte Modelle)

**Wichtig:** Verwendet die **gleiche Pipeline-Architektur**:

1. **Dieselbe Train/Test Split** (80/20 für Telco/Ecom, Temporal für Bank)
2. **Dieselbe CV-Strategie** (5-Fold Stratified für Telco/Ecom, TimeSeriesSplit für Bank)
3. **Inside-the-Loop Preprocessing** (kein Data Leakage)
4. **Gleiche Metriken** (Accuracy, F1, ROC-AUC)
5. **Vergleich gegen Baselines** (z. B. "SVM: 85% vs. LogReg Baseline: 80.55%")

**Konkrete Nächste Schritte:**
- Hyperparameter-Tuning (Grid Search, Random Search)
- Feature Importance (SHAP, Coefficients)
- Fehleranalyse (welche Instanzen misklassifiziert? Warum?)
- Ensemble-Methoden (Stacking, Voting)
- Cross-Dataset-Transfer (trainiert auf Telco, testet auf Bank?)

---

## 9. Projektstruktur (aktualisiert)

```
customer-decision-prediction/
├── .gitignore                          # data/raw, data/processed, reports/baseline_*.csv ignoriert
├── README.md
├── doc.md                              # Diese Datei
├── requirements.txt                    # (noch zu finalisieren)
│
├── data/
│   ├── raw/                            # .gitignore: nicht committed
│   │   ├── telco.csv                   # (auto-downloaded by setup_data.py)
│   │   ├── bank.csv
│   │   └── ecom.csv
│   └── processed/                      # .gitignore: nicht committed
│       ├── telco_clean.csv             # (generated by setup_data.py)
│       ├── bank_clean.csv
│       └── ecom_clean.csv
│
├── notebooks/
│   └── milestone1.ipynb                # (erstellt von der Gruppe)
│
├── reports/
│   ├── eda_report.txt                  # (generated by analysis.py)
│   ├── summary_telco.json              # (generated by analysis.py)
│   ├── summary_bank.json
│   ├── summary_ecom.json
│   ├── baseline_cv_results.csv         # .gitignore: nicht committed
│   ├── baseline_test_results.csv       # .gitignore: nicht committed
│   └── plots/
│       ├── corr_heatmap_telco.png      # (generated by analysis.py)
│       ├── corr_heatmap_bank.png
│       └── corr_heatmap_ecom.png
│
└── src/
    ├── __init__.py
    ├── setup_data.py                   # Phase 1 + 2: Download + Preprocessing
    ├── analysis.py                     # Phase 3: EDA
    ├── evaluation.py                   # Phase 4: Baseline Evaluation
    │
    └── utils/
        ├── __init__.py
        ├── download.py                 # Download utilities
        └── preprocess.py               # Preprocessing functions
```

---

## 10. Zusammenfassung und Checkliste

- ✅ **Download automatisiert:** `src/utils/download.py` cached Dateien lokal
- ✅ **Preprocessing uniform:** 7-Schritte-Pipeline, alle Datensätze gleich behandelt
- ✅ **EDA dokumentiert:** JSON-Summaries, Text-Reports, Plots
- ✅ **Baseline verglichen:** 3 Modelle, 5-Fold CV, Hold-out Test, 3 Metriken
- ✅ **Temporal Validation bei Bank:** TimeSeriesSplit, realistische Hold-out, Konzeptdrift
- ✅ **Data Leakage vermieden:** Pipelines inside CV-Loop
- ✅ **Ergebnisse maschinenlesbar:** CSV-Exports für weitere Analysen
- ✅ **Reproducible:** Random Seeds gesetzt, Skripte idempotent
- ✅ **Für Gruppe vorbereitet:** Klare Anleitung, Checklisten, nächste Schritte

**Nächste Aufgabe:** Milestone Notebook erstellen und obige Punkte diskutieren – dann geht es zu Task 3 & 4!

---

## Kontakt & Fragen

Bei Fragen zur Pipeline: Johannes  
Bei Fragen zur EDA: (Analyseteam)  
Bei Fragen zu Baseline-Interpretation: (Evaluierungsteam)
