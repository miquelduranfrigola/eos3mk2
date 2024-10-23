# imports
import os
import csv
import sys
import joblib

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.join(root, "..", "..", "checkpoints")

# loading scaler
# scaler = joblib.load(os.path.join(checkpoints_dir, "scaler.pkl"))

# loading models
model_1 = joblib.load(os.path.join(checkpoints_dir, "model1_RFC.pkl"))
model_2 = joblib.load(os.path.join(checkpoints_dir, "model2_GBC.pkl"))
model_3 = joblib.load(os.path.join(checkpoints_dir, "model3_LOGREG.pkl"))

sys.exit(0)

# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

# calculate descriptors
def descriptor_calculator(smiles_list):
    pass

descriptors = descriptor_calculator(smiles_list)

def get_index_of_descriptors_without_nans(descriptors):
    idxs = []
    for i in range(len(descriptors)):
        if np.any(np.isnan(descriptors[i,:])):
            pass
        else:
            idxs += [i]
    return idxs

idxs = get_index_of_descriptors_without_nans(descriptors)

descriptors = np.array([descriptors[i] for i in idxs])

# normalize descriptors
descriptors = scaler.transform(descriptors)

# run models
output_1 = model_1.predict_proba(descriptors)[:, 1]
output_2 = model_2.predict_proba(descriptors)[:, 1]
output_3 = model_3.predict_proba(descriptors)[:, 1]

idxs = set(idxs)

# reconstruct output
k = 0
outputs = []
for i in range(len(smiles_list)):
    if i in idxs:
        o1 = output_1[k]
        o2 = output_2[k]
        o3 = output_3[k]
        k += 1
    else:
        o1 = None
        o2 = None
        o3 = None
    outputs += [[o1, o2, o3]]

#check input and output have the same length
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len

# write output in a .csv file
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["model1_score", "model2_score", "model3_score"])  # header
    for o in outputs:
        writer.writerow([o])
