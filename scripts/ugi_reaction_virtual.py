import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolEnumerator
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from rdkit.Chem.Draw import MolDrawing
from rdkit.Chem import PandasTools
from rdkit.Chem.Scaffolds import MurckoScaffold


IPythonConsole.ipython_useSVG = True

PandasTools.RenderImagesInAllDataFrames(images=True)


def show_atom_number(
    mol, label: Literal["atomLabel", "molAtomMapNumber", "atomNote"] = "atomNote"
):
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return mol


core_smiles = "C(=O)NCC(=O)N"

# define the postion in the core for the attachment of components
core_A_pos = 2  # the position in core to attach component A, R1-NH2
core_B_pos = 6  # the position in core to attach component B, R2-NC
core_C_pos = 3  # the position in core to attach component C, R3-CHO
core_D_pos = 0  # the position in core to attach component D, R4-COOH


core_mol = Chem.MolFromSmiles(core_smiles) # Convert smile strings to rdkit molecule
core_num_atoms = core_mol.GetNumAtoms() # Get the atom numbers
show_atom_number(core_mol)


data_file = Path("SMILES_251_virtual_4CR_pretrain_data_generation_components.csv")

mol_df = pd.read_csv(data_file)

assert len(mol_df.dropna(subset=["SMILES"]).drop_duplicates(subset=["SMILES"])) == len(mol_df)

# add a column to store the molecule object
mol_df["mol"] = mol_df["SMILES"].apply(Chem.MolFromSmiles).apply(show_atom_number)

# add a column to store the molecular weight
mol_df["MW"] = mol_df["mol"].apply(Descriptors.MolWt)

mol_df["ID"] = mol_df["ID"].str.strip()
mol_df_A = mol_df[mol_df["ID"].str.startswith("A")]
mol_df_B = mol_df[mol_df["ID"].str.startswith("B")]
mol_df_C = mol_df[mol_df["ID"].str.startswith("C")]
mol_df_D = mol_df[mol_df["ID"].str.startswith("D")]





# extract the -NH2 group from mol_df_A molecules, store the index for them
frag_NH2 = Chem.MolFromSmarts("[NH2]")
mol_df_A["NH2_pos"] = mol_df_A.apply(
    lambda x: x["mol"].GetSubstructMatch(frag_NH2), axis=1
) # only return the first index


# replace the NH2 group with an open end atom *, that can later be connected to other molecules
mol_df_A["main_compoent"] = mol_df_A.apply(
    lambda x: Chem.ReplaceSubstructs(x["mol"], frag_NH2, Chem.MolFromSmiles("*"))[0],
    axis=1,
)
# renumbering
mol_df_A["main_compoent"] = mol_df_A["main_compoent"].apply(show_atom_number)

mol_df_A["main_compoent_SMILES"] = mol_df_A["main_compoent"].apply(Chem.MolToSmiles)
mol_df_A["main_num_atoms"] = mol_df_A["main_compoent"].apply(lambda x: x.GetNumAtoms())

assert len(mol_df_A.drop_duplicates(subset=["main_compoent_SMILES"])) == len(mol_df_A)



# extract the -NC group from mol_df_B molecules
frag_NC = Chem.MolFromSmarts("N#C")

mol_df_B["NC_pos"] = mol_df_B.apply(
    lambda x: x["mol"].GetSubstructMatch(frag_NC), axis=1
)

# replace the NC group with an open end atom *, that can later be connected to other molecules
mol_df_B["main_compoent"] = mol_df_B.apply(
    lambda x: Chem.ReplaceSubstructs(x["mol"], frag_NC, Chem.MolFromSmiles("*"))[0],
    axis=1,
)

# renumbering
mol_df_B["main_compoent"] = mol_df_B["main_compoent"].apply(show_atom_number)

mol_df_B["main_compoent_SMILES"] = mol_df_B["main_compoent"].apply(Chem.MolToSmiles)
mol_df_B["main_num_atoms"] = mol_df_B["main_compoent"].apply(lambda x: x.GetNumAtoms())

assert len(mol_df_B.drop_duplicates(subset=["main_compoent_SMILES"])) == len(mol_df_B)





# extract the -CHO group from mol_df_C molecules dd
frag_CHO = Chem.MolFromSmarts("[CX3H1](=O)")

mol_df_C["CHO_pos"] = mol_df_C.apply(
    lambda x: x["mol"].GetSubstructMatch(frag_CHO), axis=1
)


# replace the CHO group with an open end atom *, that can later be connected to other molecules
mol_df_C["main_compoent"] = mol_df_C.apply(
    lambda x: Chem.ReplaceSubstructs(x["mol"], frag_CHO, Chem.MolFromSmiles("*"))[0],
    axis=1,
)
# renumbering
mol_df_C["main_compoent"] = mol_df_C["main_compoent"].apply(show_atom_number)

mol_df_C["main_compoent_SMILES"] = mol_df_C["main_compoent"].apply(Chem.MolToSmiles)
mol_df_C["main_num_atoms"] = mol_df_C["main_compoent"].apply(lambda x: x.GetNumAtoms())

assert len(mol_df_C.drop_duplicates(subset=["main_compoent_SMILES"])) == len(mol_df_C)




# extract the -COOH group from mol_df_D molecules
frag_COOH = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")

mol_df_D["COOH_pos"] = mol_df_D.apply(
    lambda x: x["mol"].GetSubstructMatch(frag_COOH), axis=1
)


# replace the COOH group with an open end atom *, that can later be connected to other molecules
mol_df_D["main_compoent"] = mol_df_D.apply(
    lambda x: Chem.ReplaceSubstructs(x["mol"], frag_COOH, Chem.MolFromSmiles("*"))[0],
    axis=1,
)
# renumbering
mol_df_D["main_compoent"] = mol_df_D["main_compoent"].apply(show_atom_number)

mol_df_D["main_compoent_SMILES"] = mol_df_D["main_compoent"].apply(Chem.MolToSmiles)
mol_df_D["main_num_atoms"] = mol_df_D["main_compoent"].apply(lambda x: x.GetNumAtoms())

assert len(mol_df_D.drop_duplicates(subset=["main_compoent_SMILES"])) == len(mol_df_D)




test_A_mol = mol_df_A["main_compoent"].iloc[0]
test_B_mol = mol_df_B["main_compoent"].iloc[1]
test_C_mol = mol_df_C["main_compoent"].iloc[0]
test_D_mol = mol_df_D["main_compoent"].iloc[0]

test_A_smiles = mol_df_A["main_compoent_SMILES"].iloc[0]
test_B_smiles = mol_df_B["main_compoent_SMILES"].iloc[1]
test_C_smiles = mol_df_C["main_compoent_SMILES"].iloc[0]
test_D_smiles = mol_df_D["main_compoent_SMILES"].iloc[0]
assert test_A_smiles.startswith("*")
assert test_B_smiles.startswith("*")
assert test_C_smiles.startswith("*")
assert test_D_smiles.startswith("*")

star_of_A_pos = core_num_atoms
star_of_B_pos = core_num_atoms + test_A_mol.GetNumAtoms()
star_of_C_pos = core_num_atoms + test_A_mol.GetNumAtoms() + test_B_mol.GetNumAtoms()
star_of_D_pos = (
    core_num_atoms
    + test_A_mol.GetNumAtoms()
    + test_B_mol.GetNumAtoms()
    + test_C_mol.GetNumAtoms()
)

mol_to_combine = Chem.MolFromSmiles(
    f"{core_smiles}.{test_A_smiles}.{test_B_smiles}.{test_C_smiles}.{test_D_smiles} |m:{star_of_A_pos}:{core_A_pos},{star_of_B_pos}:{core_B_pos},{star_of_C_pos}:{core_C_pos},{star_of_D_pos}:{core_D_pos}|"
)

Draw.MolToImage(show_atom_number(mol_to_combine), size=(400, 400))


Draw.MolToImage(
    show_atom_number(rdMolEnumerator.Enumerate(mol_to_combine)[0]),
    size=(400, 400),
)


output_file = "generated_virtual_library_processed_pretraininig_data.csv"
save_every = 1000000  # Save after every 1 million molecules
combined_mols = []
count = 0

for i in range(len(mol_df_A)):
    test_A_mol, raw_A_mol, test_A_smiles, raw_A_smiles = mol_df_A.iloc[i][
        ["main_compoent", "mol", "main_compoent_SMILES", "SMILES"]
    ]
    for j in range(len(mol_df_B)):
        test_B_mol, raw_B_mol, test_B_smiles, raw_B_smiles = mol_df_B.iloc[j][
            ["main_compoent", "mol", "main_compoent_SMILES", "SMILES"]
        ]
        for k in range(len(mol_df_C)):
            test_C_mol, raw_C_mol, test_C_smiles, raw_C_smiles = mol_df_C.iloc[k][
                ["main_compoent", "mol", "main_compoent_SMILES", "SMILES"]
            ]
            for l in range(len(mol_df_D)):
                test_D_mol, raw_D_mol, test_D_smiles, raw_D_smiles = mol_df_D.iloc[l][
                    ["main_compoent", "mol", "main_compoent_SMILES", "SMILES"]
                ]
                if count % 10 == 0:
                    print(f"Generating the No.{count} molecule")

                assert test_A_smiles.startswith("*")
                assert test_B_smiles.startswith("*")
                assert test_C_smiles.startswith("*")
                assert test_D_smiles.startswith("*")

                star_of_A_pos = core_num_atoms
                star_of_B_pos = core_num_atoms + test_A_mol.GetNumAtoms()
                star_of_C_pos = (
                    core_num_atoms + test_A_mol.GetNumAtoms() + test_B_mol.GetNumAtoms()
                )
                star_of_D_pos = (
                    core_num_atoms
                    + test_A_mol.GetNumAtoms()
                    + test_B_mol.GetNumAtoms()
                    + test_C_mol.GetNumAtoms()
                )

                mol_to_combine = Chem.MolFromSmiles(
                    f"{core_smiles}.{test_A_smiles}.{test_B_smiles}.{test_C_smiles}.{test_D_smiles} |m:{star_of_A_pos}:{core_A_pos},{star_of_B_pos}:{core_B_pos},{star_of_C_pos}:{core_C_pos},{star_of_D_pos}:{core_D_pos}|"
                )

                enumerated = rdMolEnumerator.Enumerate(mol_to_combine)
                assert len(enumerated) == 1
                combined_mol = enumerated[0]

                results = {
                    "id": count,
                    "combined_mol": combined_mol,
                    "combined_mol_SMILES": Chem.MolToSmiles(combined_mol),
                    "A": raw_A_mol,
                    "A_smiles": raw_A_smiles,
                    "B": raw_B_mol,
                    "B_smiles": raw_B_smiles,
                    "C": raw_C_mol,
                    "C_smiles": raw_C_smiles,
                    "D": raw_D_mol,
                    "D_smiles": raw_D_smiles,
                }

                combined_mols.append(results)
                count += 1

                # Save every 1 million molecules
                if count % save_every == 0:
                    df = pd.DataFrame(combined_mols)

                    # Check if file exists to determine whether to write header
                    write_header = not os.path.exists(output_file)
                    df.to_csv(output_file, mode='a', index=False, header=write_header)

                    print(f"Saved {count} molecules to {output_file}")
                    combined_mols.clear()  # Free memory after saving

# Save any remaining molecules at the end
if combined_mols:
    df = pd.DataFrame(combined_mols)
    write_header = not os.path.exists(output_file)
    df.to_csv(output_file, mode='a', index=False, header=write_header)
    print(f"Final save: {count} molecules written to {output_file}")

