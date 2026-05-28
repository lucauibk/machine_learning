# Präsentation Vorbereitung — Team Overfitters

5 Minuten, Proseminar 25./26. Juni 2026. Beide müssen alles erklären können.

---

## Das Projekt

Wir lösen ein **Regressionsproblem**: Aus Wetterdaten von fünf Städten (Bedrock, Gotham City, New New York, Paperopoli, Springfield) sagen wir den Gesamtstromverbrauch eines fiktiven Stromnetzes in Megawatt vorher.

Der Datensatz hat **39.997 Zeilen** und **65 Features** — 55 davon sind Zahlen (Temperatur, Luftdruck, Luftfeuchtigkeit, Windgeschwindigkeit, Windrichtung, Niederschlag, Schnee, Wolken pro Stadt), 10 sind kategorisch (Wetterkategorie und Beschreibungstext pro Stadt). Es gibt **keine fehlenden Werte**.

Da kein separater Test-Split vorhanden ist, haben wir selbst aufgeteilt: **80% Training (31.997 Samples), 20% Validation (8.000 Samples)**, mit `random_state=42` damit die Ergebnisse reproduzierbar sind.

---

## Preprocessing

### weather_description gedroppt
Die 5 `weather_description`-Spalten enthalten Freitext wie "light rain" oder "broken clouds". Diese Information steckt bereits vollständig in den 5 `weather_main`-Spalten (z.B. "Rain", "Clouds") — nur mit weniger Kategorien und ohne Redundanz. Deshalb haben wir die Beschreibungsspalten entfernt. Es ist kein Informationsverlust, nur Vereinfachung.

### StandardScaler auf numerische Features
Ridge Regression und MLP bestrafen große Gewichte bzw. sind empfindlich gegenüber unterschiedlichen Feature-Skalen. Temperatur liegt z.B. um ~280 K, Windrichtung zwischen 0 und 360, Wolkendecke zwischen 0 und 1. Ohne Skalierung würden Features mit großen Werten die Gewichte dominieren — nicht weil sie informativer sind, sondern weil sie größer sind. Der `StandardScaler` zieht von jedem Feature den Mittelwert ab und teilt durch die Standardabweichung → alle Features haben danach Mittelwert 0 und Standardabweichung 1.

### OneHotEncoder auf weather_main
Die 5 `weather_main`-Spalten enthalten Textkategorien wie "Rain", "Clear", "Thunderstorm". Ein Modell kann mit Text nicht rechnen. Der `OneHotEncoder` wandelt jede Kategorie in eine eigene 0/1-Spalte um. Bei 9 eindeutigen Kategorien pro Stadt und 5 Städten entstehen 45 neue binäre Features.

**Endergebnis:** 55 skalierte numerische + 45 OHE-Features = **100 Features**.

### Kein Leakage
Der Preprocessor (Scaler + Encoder) wird **ausschließlich auf den Trainingsdaten gefittet** — also Mittelwert, Standardabweichung und Kategorien werden nur aus den 31.997 Trainingssamples berechnet. Auf die Validierungsdaten wird der bereits gefittete Preprocessor nur angewendet (transform, nicht fit). Würde man den Scaler auf allen Daten fitten, würde Information aus der Validation in das Training fließen — das nennt sich Data Leakage und würde die Validierungsergebnisse zu gut erscheinen lassen.

---

## Method 1: Ridge Regression

### Was es ist
Ridge Regression ist lineare Regression mit L2-Regularisierung. Sie minimiert:

```
ŵ = argmin_w  ‖Xw - y‖²  +  α‖w‖²
```

Der erste Term ist der normale Vorhersagefehler (least squares). Der zweite Term `α‖w‖²` bestraft große Gewichte. Das entspricht einer MAP-Schätzung mit einem Gauß-Prior auf den Gewichten, wie in Vorlesung 4 hergeleitet.

### Was α macht
`α` steuert den Bias-Variance Trade-off:
- **Großes α**: Gewichte werden stark bestraft → alle Gewichte gehen gegen 0 → Modell ist sehr einfach → Underfitting (hoher Bias)
- **Kleines α**: Kaum Bestrafung → fast identisch mit normaler linearer Regression (OLS) → Modell kann Gewichte frei wählen → potenziell Overfitting bei vielen Features

### Wie wir α gewählt haben
Wir haben **60 Werte logarithmisch gleichverteilt** zwischen 0.001 und 10.000 ausprobiert. Für jeden Wert haben wir ein Modell auf den Trainingsdaten trainiert und den RMSE auf den Validierungsdaten berechnet. Das α mit dem **niedrigsten Validation-RMSE** wurde gewählt.

Das beste α war **0.001** — quasi keine Regularisierung. Das bedeutet: mit 100 gut vorverarbeiteten Features ist das Problem gut gestellt, es braucht keinen starken Schrumpfungseffekt.

### Ergebnis
- Train RMSE: 3909 MW, Train R²: 0.317
- Val RMSE: 3950 MW, Val R²: 0.310
- Train und Validation sind fast gleich → kein Overfitting, aber das Modell kann die Varianz im Stromverbrauch nur zu 31% erklären. Das liegt daran, dass der Zusammenhang zwischen Wetter und Stromverbrauch nicht-linear ist — Ridge kann das nicht modellieren.

---

## Method 2: MLP Neural Network

### Was es ist
Ein Multi-Layer Perceptron (MLP) ist ein vollständig verbundenes neuronales Netz:
- **Input-Layer**: 100 Features
- **Hidden Layer(s)**: Neuronen mit nicht-linearer Aktivierungsfunktion
- **Output-Layer**: 1 Neuron (Stromverbrauch in MW)

Jedes Neuron berechnet: `Aktivierung(w · x + b)`. Durch mehrere Schichten kann das Netz beliebig komplexe nicht-lineare Zusammenhänge lernen.

### Training: Backpropagation + Adam
Das Netz wird mit **Backpropagation** trainiert: Der Vorhersagefehler (MSE) wird am Output berechnet und via Kettenregel schichtweise rückwärts durch das Netz propagiert, um die Gradienten jedes Gewichts zu berechnen. Der **Adam-Optimizer** nutzt diese Gradienten + adaptiv angepasste Lernraten je Gewicht, um die Gewichte zu aktualisieren.

### Aktivierungsfunktion tanh
`tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})`, Ausgabe zwischen -1 und 1. Im Vergleich zu ReLU (das für negative Eingaben immer 0 liefert) hat tanh überall einen Gradienten ungleich null — das kann das Training bei diesem Datensatz stabilisieren. Bei unserer Random Search hat tanh besser abgeschnitten als ReLU.

### Hyperparameter-Suche
Wir haben **Random Search über 30 Konfigurationen** durchgeführt. Dabei wurden zufällig kombiniert:
- Hidden Layer Größen: (64), (128), (256), (64,32), (128,64), (256,128), (128,64,32)
- Learning Rates: 0.0001, 0.0003, 0.001, 0.003, 0.01
- Aktivierungen: relu, tanh

Jedes Modell wurde 200 Epochen trainiert, dann wurde der Validation-RMSE berechnet. Die beste Konfiguration wurde ausgewählt.

### Beste Konfiguration
- 1 Hidden Layer mit **256 Neuronen**
- Aktivierung: **tanh**
- Learning Rate: **0.01**
- Max Epochen: **200**

### Ergebnis
- Train RMSE: 2742 MW, Train R²: 0.664
- Val RMSE: 3139 MW, Val R²: 0.564
- Der Gap zwischen Train R²=0.664 und Val R²=0.564 zeigt **mildes Overfitting** — das Modell hat Trainingsdaten etwas auswendig gelernt. Könnte mit Weight Decay oder Dropout reduziert werden.
- Deutlich besser als Ridge: R² 0.564 vs. 0.310 → nicht-lineare Zusammenhänge zwischen Wetter und Stromverbrauch sind real und bedeutsam.

---

## Gespeichertes Modell + Test-Notebook

Das beste Modell (MLP) ist als `models/final_model.pkl` gespeichert (**0.87 MB**, weit unter dem 50 MB Limit). Das PKL-File enthält ein scikit-learn `Pipeline`-Objekt mit zwei Schritten:
1. `pre`: der gefittete `ColumnTransformer` (Scaler + OHE)
2. `model`: das trainierte `MLPRegressor`-Objekt

Im Test-Notebook (`jupyter_abgabe.ipynb`) wird das Modell nur **geladen und angewendet, nie trainiert**:

```python
def leader_board_predict_fn(values):
    import joblib
    model = joblib.load("models/final_model.pkl")
    return model.predict(values)
```

`model.predict(values)` führt automatisch erst das Preprocessing (Scaler + OHE) und dann die MLP-Vorhersage aus — alles in einem Schritt, weil beides in der Pipeline steckt.

**Warum kein Training im Test-Notebook?** Laut Aufgabenstellung ist das explizit verboten (50% Abzug). Die Idee: Der Betreuer testet das Modell auf unseen Test-Daten ohne es neu zu trainieren — nur so ist die Evaluation fair.

---

## Was hätte man noch versuchen können

- **Temporale Features**: Stunde des Tages, Wochentag, Monat fehlen im Datensatz — aber Stromverbrauch hat starke Tages- und Jahressaisonalität. Das wäre mit Abstand der größte Hebel für bessere Vorhersagen.
- **Overfitting reduzieren**: MLP-Parameter `alpha` (Weight Decay) oder `early_stopping=True`
- **Gradient Boosting** (z.B. `HistGradientBoostingRegressor`): funktioniert auf tabellarischen Daten oft besser als MLP, würde als dritte Methode passen
- **k-NN Regression**: wurde verworfen wegen Curse of Dimensionality — in 100 Dimensionen werden alle Punkte ähnlich weit voneinander entfernt, k-NN funktioniert schlecht
