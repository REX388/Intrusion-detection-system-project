# Intrusion Detection Systems [![forthebadge made-with-python 2](https://img.shields.io/badge/Made%20with%20-Python%202.7-brightgreen.svg)](https://www.python.org/) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]() [![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)]() 
This repo consists of all the codes and datasets of the research paper, **"[Evaluating Shallow and Deep Neural Networks for Network Intrusion Detection Systems in Cyber Security](https://ieeexplore.ieee.org/document/8494096)"**.


## Abstract :
Intrusion detection system (IDS) has become an essential layer in all the latest ICT system due to an urge towards cyber safety in the day-to-day world. Reasons including uncertainty in ﬁnding the types of attacks and increased the complexity of advanced cyber attacks, IDS calls for the need of integration of Deep Neural Networks (DNNs). In this paper, DNNs have been utilized to predict the attacks on Network Intrusion Detection System (N-IDS). A DNN with 0.1 rate of learning is applied and is run for 1000 number of epochs and KDDCup-’99’ dataset has been used for training and benchmarking the network. For comparison purposes, the training is done on the same dataset with several other classical machine learning algorithms and DNN of layers ranging from 1 to 5. The results were compared and concluded that a DNN of 3 layers has superior performance over all the other classical machine learning algorithms. 

## Keywords : 
Intrusion detection, deep neural networks, machine learning, deep learning 

## Authors :
**[Rahul-Vigneswaran K](https://rahulvigneswaran.github.io)**<sup>∗</sup>, [Vinayakumar R](https://scholar.google.co.in/citations?user=oIYw0LQAAAAJ&hl=en&oi=ao)<sup>†</sup>, [Soman KP](https://scholar.google.co.in/citations?user=R_zpXOkAAAAJ&hl=en)<sup>†</sup> and [Prabaharan Poornachandran](https://scholar.google.com/citations?user=e233m6MAAAAJ&hl=en)<sup>‡</sup> 

**<sup>∗</sup>Department of Mechanical Engineering, Amrita Vishwa Vidyapeetham, India.** <br/> 
<sup>†</sup>Center for Computational Engineering and Networking (CEN), Amrita School of Engineering, Coimbatore.<br/> 
<sup>‡</sup>Center for Cyber Security Systems and Networks, Amrita School of Engineering, Amritapuri Amrita Vishwa Vidyapeetham, India.

## CYB 213 Assignment: Real-Time IDS Simulation

This assignment implements a **Real-Time Intrusion Detection System (IDS)** using Logistic Regression for binary classification (Normal vs. Attack). The project trains a model on 5% of the KDD Cup '99 dataset and simulates real-time streaming alert generation.

---

## Project Structure

```
.
├── all.py                          # Train classical ML models (LR, NB, KNN, DT, AB, RF, SVM)
├── realtime/
│   └── realtime_ids.py             # Real-time IDS simulator
├── classical/                      # Stores model outputs and predictions
│   ├── logistic_model.joblib       # Saved Logistic Regression model
│   ├── normalizer.joblib           # Saved data normalizer/scaler
│   └── *.txt                       # Prediction outputs
├── dnn/                            # Deep Neural Network (100 epochs)
├── dnn1000/                        # Deep Neural Network (1000 epochs)
├── kddtrain.csv                    # Training dataset (ignored by .gitignore)
├── kddtest.csv                     # Test dataset (ignored by .gitignore)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Installation

### Prerequisites
- Python 3.7+
- pip or conda

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy pandas scikit-learn joblib
```

---

## How to Run

### Step 1: Train the Model and Save

Run the training script to:
- Load the KDD Cup '99 dataset (uses 5% for faster execution)
- Train 8 classical ML models (Logistic Regression, Naive Bayes, KNN, Decision Tree, AdaBoost, Random Forest, SVM-rbf, SVM-linear)
- Save the Logistic Regression model and normalizer
- Output evaluation metrics (accuracy, precision, recall, F1-score) and predictions

```bash
python all.py
```

**Output:**
- Trained model saved to `classical/logistic_model.joblib`
- Scaler/normalizer saved to `classical/normalizer.joblib`
- Predictions and metrics saved to `classical/*.txt` files
- Console prints accuracy, precision, recall, F1-score for each model

---

### Step 2: Run Real-Time IDS Simulation

Simulate live intrusion detection streaming. The script:
- Loads the saved Logistic Regression model and normalizer
- Streams through test data samples (5% subset)
- Classifies each sample and prints timestamped "Normal" or "Attack" alerts to console
- Simulates realistic timing with 0.5-second delays between samples

```bash
python .\realtime\realtime_ids.py
```

**Output Example:**
```
--- Real-Time IDS Simulation (5% of test data) ---
[2025-12-03 10:30:45] Sample 1: Normal
[2025-12-03 10:30:46] Sample 2: Attack
[2025-12-03 10:30:47] Sample 3: Normal
...
```

---

## Code Explanation

### `all.py` – Model Training and Evaluation

**Key Sections:**

1. **Imports & Setup** (Lines 1–21)
   - Imports scikit-learn models, metrics, and joblib for model persistence
   
2. **Data Loading & Preprocessing** (Lines 23–48)
   - Loads KDD Cup '99 training and test data as CSV
   - Selects 5% of data for efficiency: `train_frac = int(0.05 * len(traindata))`
   - Splits features (columns 1–41) and labels (column 0)
   - Normalizes features using `Normalizer().fit(X)` to scale to unit norm
   
3. **Logistic Regression Training** (Lines 57–67)
   - Creates and trains LogisticRegression model
   - Saves model and scaler for later use:
     ```python
     dump(model, 'classical/logistic_model.joblib')
     dump(scaler, 'classical/normalizer.joblib')
     ```
   - Evaluates on test set and prints accuracy, precision, recall, F1-score
   
4. **Other Classical Models** (Lines 69–308)
   - Trains and evaluates: Naive Bayes, KNN, Decision Tree, AdaBoost, Random Forest, SVM-rbf, SVM-linear
   - Same evaluation metrics printed for each
   - Predictions saved to `.txt` files for later analysis

---

### `realtime/realtime_ids.py` – Real-Time IDS Simulator

**Key Sections:**

1. **Load Pre-trained Model** (Lines 6–10)
   - Uses `os.path.dirname()` to resolve file paths relative to script location
   - Loads saved Logistic Regression model and normalizer from `classical/`
   
2. **Prepare Streaming Data** (Lines 13–21)
   - Reads test CSV file
   - Uses 5% of test data to simulate streaming
   - Extracts features (columns 1–41) and true labels (column 0)
   - Normalizes features using saved scaler: `T_scaled = scaler.transform(T)`
   
3. **Real-Time Classification & Alert** (Lines 23–28)
   - Iterates through each sample in the stream
   - Classifies using saved model: `pred = model.predict(sample.reshape(1, -1))[0]`
   - Maps predictions to labels: `'Normal'` if pred=0, else `'Attack'`
   - Prints timestamped alert: `[HH:MM:SS] Sample N: {Normal|Attack}`
   - Sleeps 0.5 seconds to simulate real-time stream timing

---

## Assignment Requirements (CYB 213)

✅ **Requirement** → **Implementation**

| Requirement | Implementation |
|---|---|
| **Python-based** | Pure Python with numpy, pandas, scikit-learn, joblib |
| **Allowed libraries only** | No external/prohibited packages used |
| **Runs on Colab/Jupyter/VS Code** | Plain scripts; easily portable to notebooks |
| **Logistic Regression algorithm** | ✅ Trained in `all.py` |
| **Real-time streaming IDS** | ✅ Implemented in `realtime_ids.py` |
| **Console "Normal" / "Attack" alerts** | ✅ Timestamped output in real-time script |
| **Uses 5% dataset** | ✅ Lines 27–30 in `all.py` |
| **Max 30 marks** | Project scope within assignment bounds |

---

## Dataset

- **Source:** KDD Cup '99 Intrusion Detection Dataset
- **Features:** 41 attributes per sample
- **Labels:** Binary (Normal = 0, Attack = 1)
- **Size used:** 5% of full dataset (for faster execution on resource-constrained systems)

---

## Performance Metrics

Each model in `all.py` outputs:
- **Accuracy:** Overall correctness (TP + TN) / Total
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1-Score:** Harmonic mean of precision and recall

---

## Dependencies

See `requirements.txt`:
- `numpy` – Numerical computations
- `pandas` – Data loading and manipulation
- `scikit-learn` – Machine learning models and metrics
- `joblib` – Model serialization/deserialization

---

## Files to Submit

For grading, include:
- `all.py` – Training script
- `realtime/realtime_ids.py` – Real-time simulator
- `requirements.txt` – Dependencies
- `README.md` – This documentation

(Data files `*.csv` and model files `*.joblib` are excluded via `.gitignore` to reduce repo size.)

---

## How to Run the Code (Original Section)
### For **Classical Machine Learning**
* Run `all.py` to train and evaluate 8 classical ML models (Logistic Regression, Naive Bayes, KNN, Decision Tree, AdaBoost, Random Forest, SVM-rbf, SVM-linear).

### For **Real-Time IDS Simulation**
* Run `realtime/realtime_ids.py` to simulate live intrusion detection alerts (uses Logistic Regression trained model).

### For **Deep Neural Network (100 iterations)** 
* Run `dnn1.py` for 1-hidden layer network and run `dnn1acc.py` for finding it's accuracy.
* Run `dnn2.py` for 2-hidden layer network and run `dnn2acc.py` for finding it's accuracy.
* Run `dnn3.py` for 3-hidden layer network and run `dnn3acc.py` for finding it's accuracy.
* Run `dnn4.py` for 4-hidden layer network and run `dnn4acc.py` for finding it's accuracy.
* Run `dnn5.py` for 5-hidden layer network and run `dnn5acc.py` for finding it's accuracy.

### For **Deep Neural Network (1000 iterations)** 
* Run `dnn1.py` for 1-hidden layer network and run `dnn1acc.py` for finding it's accuracy.
* Run `dnn2.py` for 2-hidden layer network and run `dnn2acc.py` for finding it's accuracy.
* Run `dnn3.py` for 3-hidden layer network and run `dnn3acc.py` for finding it's accuracy.
* Run `dnn4.py` for 4-hidden layer network and run `dnn4acc.py` for finding it's accuracy.
* Run `dnn5.py` for 5-hidden layer network and run `dnn5acc.py` for finding it's accuracy.



## Recommended Citation :
If you use this repository in your research, cite the the following papers :

  1. Rahul, V.K., Vinayakumar, R., Soman, K.P., & Poornachandran, P. (2018). Evaluating Shallow and Deep Neural Networks for Network Intrusion Detection Systems in Cyber Security. 2018 9th International Conference on Computing, Communication and Networking Technologies (ICCCNT), 1-6.
  2. Rahul-Vigneswaran, K., Poornachandran, P., & Soman, K.P. (2019). A Compendium on Network and Host based Intrusion Detection Systems. CoRR, abs/1904.03491.
  
  ### Bibtex Format :
```bib
@article{Rahul2018EvaluatingSA,
  title={Evaluating Shallow and Deep Neural Networks for Network Intrusion Detection Systems in Cyber Security},
  author={Vigneswaran K Rahul and R. Vinayakumar and K. P. Soman and Prabaharan Poornachandran},
  journal={2018 9th International Conference on Computing, Communication and Networking Technologies (ICCCNT)},
  year={2018},
  pages={1-6}
  }

@article{RahulVigneswaran2019ACO,
  title={A Compendium on Network and Host based Intrusion Detection Systems},
  author={K Rahul-Vigneswaran and Prabaharan Poornachandran and K. P. Soman},
  journal={CoRR},
  year={2019},
  volume={abs/1904.03491}
  }
```

## Issue / Want to Contribute ? :
Open a new issue or do a pull request incase your are facing any difficulty with the code base or you want to contribute to it.

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)]()

