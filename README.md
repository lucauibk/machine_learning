# Machine Learning Projekt — Team Overfitters

Einführung in Machine Learning, SS 2026  
**Aufgabe:** Regression — Gesamtstromverbrauch (MW) aus Wetterdaten von 5 Städten vorhersagen

---

## Projektstruktur

```
project/
├── training.ipynb              # Haupttraining: EDA, Preprocessing, Ridge + MLP, Figures
├── jupyter_abgabe.ipynb        # Abgabe-Notebook für JupyterHub (lädt Modell, kein Training)
├── models/                     # Gespeicherte Modelle (von Git ignoriert)
│   ├── final_model.pkl         # Bestes Modell (MLP), wird im Abgabe-Notebook geladen
│   ├── mlp_model.pkl           # MLP Pipeline (Preprocessor + MLPRegressor)
│   ├── ridge_model.pkl         # Ridge Pipeline
│   ├── mlp_search.pkl          # Ergebnisse der MLP Random Search (30 Trials)
│   └── ridge_search.pkl        # Ergebnisse der Ridge Grid Search (60 Alpha-Werte)
├── figures/                    # Generierte Plots (von Git ignoriert)
│   ├── fig1_data_overview.png  # Verteilung Stromverbrauch + Temperatur vs. Verbrauch
│   ├── fig2_ridge_hyperparameter.png  # RMSE vs. Alpha (Ridge Grid Search)
│   ├── fig3_mlp_loss_curve.png        # Trainings-Loss-Kurve des besten MLP
│   └── fig4_predicted_vs_actual.png   # Predicted vs. Actual für beide Modelle
└── report/
    ├── report.tex              # LaTeX-Bericht (ieeeconf-Template)
    ├── report.pdf              # Kompilierter Bericht (von Git ignoriert)
    └── ieeeconf.cls            # LaTeX-Klasse für das offizielle Template

instructions_examples/
├── project-instructions.pdf   # Offizielle Aufgabenstellung
├── Datasets/
│   └── powerpredict.csv       # Datensatz (39.997 Samples, 65 Features)
├── Example Report/
│   └── report.pdf             # Beispiel-Bericht vom Betreuer
└── Report Template/
    └── report_latex_template.zip  # Offizielles LaTeX-Template
```

---

## Notebooks

### `project/training.ipynb`
Vollständige ML-Pipeline:
1. Daten laden und explorieren
2. Preprocessing (StandardScaler + OneHotEncoder)
3. Ridge Regression mit Grid Search über 60 Alpha-Werte
4. MLP Neural Network mit Random Search über 30 Konfigurationen
5. Vergleich beider Modelle, Figures generieren
6. Bestes Modell als `final_model.pkl` speichern

Mit `TRAIN = False` (Zeile 2) wird das Training übersprungen und die gespeicherten Modelle geladen.

### `project/jupyter_abgabe.ipynb`
Abgabe-Notebook für JupyterHub. Lädt `models/final_model.pkl` und macht Vorhersagen.  
**Kein Training** — die Funktion `leader_board_predict_fn` ruft nur `model.predict()` auf.  
`get_score()` berechnet den MAE auf dem Trainings- und (falls vorhanden) dem versteckten Test-Datensatz.

---

## ML Pipeline

| Schritt | Details |
|---------|---------|
| Features | 55 numerisch + 5 kategorisch (`weather_main`) nach Drop von `weather_description` |
| Preprocessing | `StandardScaler` auf numerische, `OneHotEncoder` auf kategorische → 100 Features |
| Split | 80/20 Train/Validation, `random_state=42` |
| Method 1 | Ridge Regression, Grid Search über 60 Alpha-Werte in [1e-3, 1e4] |
| Method 2 | MLP, Random Search über 30 Configs (Hidden Size, LR, Activation) |
| Bestes Modell | MLP: Val RMSE 3139 MW, Val R² 0.564 |

---

## Bericht kompilieren

```bash
cd project/report && pdflatex report.tex
```

---

## Deadlines

| Was | Wann |
|-----|------|
| Code-Abgabe (ZIP + JupyterHub) | 24. Juni 2026, 12:00 Uhr |
| Code-Präsentation | 25./26. Juni 2026 |
| Bericht (PDF auf OLAT) | 9. Juli 2026, 23:59 Uhr |
