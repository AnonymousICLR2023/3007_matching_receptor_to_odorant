from rdkit import Chem
try:
    from chembl_structure_pipeline import standardizer
    isavailable_chembl_structure_pipeline = True
except ImportError:
    print('chembl_structure_pipeline is not installed. Can not use following functions:')
    print('standardizer')
    isavailable_chembl_structure_pipeline = False       



def validate_atom_features():
    pass

def validate_bond_features():
    pass


def get_num_atoms_and_bonds(smiles, IncludeHs = False, validate = False):
    mol = Chem.MolFromSmiles(smiles.strip())
    assert mol is not None

    if IncludeHs:
        mol = Chem.rdmolops.AddHs(mol)

    if validate:
        can_smi = Chem.MolToSmiles(mol) # canonical SMILES - TO DO: Check if this row is necessary.
        raise NotImplementedError("validate = True is not implemented at the moment. Use validate = False.")

    return len(mol.GetAtoms()), len(mol.GetBonds())


if isavailable_chembl_structure_pipeline:
    def standardize_smiles(smi):
        smimol = Chem.MolFromSmiles(smi)
        parent, flag = standardizer.get_parent_mol(smimol)
        newmol = standardizer.standardize_mol(parent)
        smiles = Chem.MolToSmiles(newmol)
        return smiles


if __name__ == '__main__':
    smiles = 'C/C/1=C/CC/C(=C\[C@H]2[C@H](C2(C)C)CC1)/C'
    print(get_num_atoms(smiles))