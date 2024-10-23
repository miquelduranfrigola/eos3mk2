import os
import pandas as pd
import joblib
import sys
import numpy as np
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "..", "code"))
from charge_neutralizer import NeutraliseCharges
from descriptors_calculator import descriptors_calculator

print("Loading the molecules")
df = pd.read_excel(os.path.join(root, "datasetsCompounds.xlsx"), sheet_name="CPSMs")
print(df.head())

def str2float(x):
    try:
        return float(x)
    except:
        return np.nan

y_logbb = np.array([str2float(x) for x in df["logBB value"].tolist()])

smiles_list = []
for smi in df["SMILES"].tolist():
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        smiles_list += [None]
    else:
        smiles_list += [Chem.MolToSmiles(mol)]

print("Neutralizing SMILES")
neutralized_smiles = []
for smiles in smiles_list:
    neutralized_smiles += [NeutraliseCharges(smiles)[0]]

print("Calculating descriptors")
descriptors = np.array(descriptors_calculator(neutralized_smiles))

# MinMax scaling
scaler = MinMaxScaler()
scaler.fit(descriptors)

# Save scaler
joblib.dump(scaler, os.path.join(root, "..", "..", "checkpoints", "scaler.joblib"))

# Scale descriptors
descriptors_scaled = np.array(pd.DataFrame(scaler.transform(descriptors)).fillna(0))

# Binarize classification
y_logbb_class = []
for y in y_logbb:
    if np.isnan(y):
        y_logbb_class += [np.nan]
        continue
    if y > 0.1:
        y_logbb_class += [1]
    else:
        y_logbb_class += [0]
y_logbb_class = np.array(y_logbb_class)
print(y_logbb_class)

# Get data
mask = ~np.isnan(y_logbb_class)
x = descriptors_scaled[mask]
y = y_logbb_class[mask]

# Setting up for ML
from sklearn import model_selection
seed = 42
skfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed, shuffle= True)

# Models
from sklearn.ensemble import RandomForestClassifier
opt_rfc = RandomForestClassifier(n_estimators=700, max_depth=5, max_features='log2', min_samples_leaf = 2)
scores = model_selection.cross_val_score(opt_rfc, x, y, cv=skfold, scoring='accuracy')
print("Stratified 10-fold CV Model accuracy %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std()*100))
opt_rfc.fit(x, y)
joblib.dump(opt_rfc, os.path.join(root, "..", "..", "checkpoints", "model1_rfc.joblib"))

from sklearn.ensemble import GradientBoostingClassifier
opt_gbc = GradientBoostingClassifier(n_estimators=300, max_depth=15, max_features='log2', min_samples_leaf = 4)
scores = model_selection.cross_val_score(opt_gbc, x, y, cv=skfold, scoring='accuracy')
print("Stratified 10-fold CV Model accuracy %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std()*100))
opt_gbc.fit(x, y)
joblib.dump(opt_gbc, os.path.join(root, "..", "..", "checkpoints", "model2_gbc.joblib"))

from sklearn.linear_model import LogisticRegression
opt_logreg = LogisticRegression(penalty = 'l2', C = 10)
scores = model_selection.cross_val_score(opt_logreg, x, y, cv=skfold, scoring='accuracy')
print("Stratified 10-fold CV Model accuracy %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std()*100))
opt_logreg.fit(x, y)
joblib.dump(opt_logreg, os.path.join(root, "..", "..", "checkpoints", "model3_logreg.joblib"))