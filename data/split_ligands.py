"""Split actives and decoys .mol2 file into individual molecule files."""
import os
import numpy as np


# Function definitions
def get_mol_name(mol_file):
    return mol_file.split("\n")[0] + ".mol2"

def get_split_files(path):
    SPLIT_ON = "@<TRIPOS>MOLECULE\n"
    with open(path, "r") as f:
        text = f.read()
    text = text.split(SPLIT_ON)[1:]
    fnames = [get_mol_name(mol_file) for mol_file in text]
    text = [SPLIT_ON + mol_file for mol_file in text]
    return text, fnames

def write_mol_files(mol_text, fname):
    with open(fname, "w") as f:
        f.write(mol_text)

def write_mol_folder(directory, text, fnames):
    try:
        os.mkdir(directory)
    except:
        pass
    for mol_text, fname in zip(text, fnames):
        write_mol_files(mol_text, f"{directory}/{fname}")

def write_split_files(sample_to=1):
    PATHS = ("actives_final.mol2", "decoys_final.mol2")
    text, fnames = get_split_files("actives_final.mol2")
    write_mol_folder("actives", text, fnames)

    num_decoys = len(fnames) * sample_to
    text, fnames = get_split_files("decoys_final.mol2")
    np.random.seed(12345)
    selected_indices = np.random.choice(range(len(fnames)), num_decoys,
                                        replace=False)
    write_mol_folder("decoys", (np.array(text)[selected_indices]).tolist(),
                        (np.array(fnames)[selected_indices]).tolist())

# Application
write_split_files()
