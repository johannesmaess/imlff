
from schnetpack.datasets import MD17
from schnetpack.datasets.md22 import MD22

from imlff.data.cumulene import Cumulene


md17_conformations_per_molecule = {
    'aspirin': 211762,
    'azobenzene': 99999,
    'benzene': 627983,
    'ethanol': 555092,
    'malonaldehyde': 993237,
    'naphthalene': 326250,
    'paracetamol': 106490,
    'salicylic_acid': 320231,
    'toluene': 442790,
    'uracil': 133770
}
md22_conformations_per_molecule = {
    'Ac-Ala3-NHMe': 85109,
    'DHA': 69753,
    'stachyose': 27272,
    'AT-AT': 20001,
    'AT-AT-CG-CG': 10153,
    'buckyball-catcher': 6102,
    'double-walled_nanotube': 5032
}

def molecule_name_typed(molecule):
    return {
        'aspirin': 'Aspirin',
        'azobenzene': 'Azobenzene',
        'benzene': 'Benzene',
        'ethanol': 'Ethanol',
        'malonaldehyde': 'Malonaldehyde',
        'naphthalene': 'Naphthalene',
        'paracetamol': 'Paracetamol',
        'salicylic_acid': 'Salicylic Acid',
        'toluene': 'Toluene',
        'uracil': 'Uracil',
        'Ac-Ala3-NHMe': 'Ac-Ala3-NHMe',
        'DHA': 'DHA',
        'stachyose': 'Stachyose',
        'AT-AT': 'AT-AT',
        'AT-AT-CG-CG': 'AT-AT-CG-CG',
        'buckyball-catcher': 'Buckyball-Catcher',
        'double-walled_nanotube': 'Double-Walled Nanotube'
    }[molecule]


cumulene_conformations_per_molecule = {
    'cumulene1_train': 4999,
    'cumulene2_train': 4998,
    'cumulene3_train': 4993,
    'cumulene4_train': 4987,
    'cumulene5_train': 4990,
    'cumulene6_train': 4975,
    'cumulene7_train': 4988,
    'cumulene8_train': 4973,
    
    'cumulene1_test': 1000,
    'cumulene2_test': 1000,
    'cumulene3_test': 1000,
    'cumulene4_test': 1000,
    'cumulene5_test': 1000,
    'cumulene6_test': 1000,
    'cumulene7_test': 1000,
    'cumulene8_test': 1000,
    
    'cumulene1_dft_train': 4999,
    'cumulene2_dft_train': 4998,
    'cumulene3_dft_train': 4993,
    'cumulene4_dft_train': 4987,
    'cumulene5_dft_train': 4990,
    'cumulene6_dft_train': 4975,
    'cumulene7_dft_train': 4988,
    'cumulene8_dft_train': 4973,
}

def molecule_to_dataset(molecule, name=True):
    if molecule in md17_conformations_per_molecule:
        Ds = MD17
    elif molecule in md22_conformations_per_molecule:
        Ds = MD22
    elif molecule in cumulene_conformations_per_molecule:
        Ds = Cumulene
    else:
        raise ValueError(f"Unknown molecule: {molecule}")
      
    if name:
        return Ds.__name__
    return Ds  

def molecule_to_n_conformations(molecule):
    if molecule in md17_conformations_per_molecule:
        return md17_conformations_per_molecule[molecule]
    if molecule in md22_conformations_per_molecule:
        return md22_conformations_per_molecule[molecule]
    if molecule in cumulene_conformations_per_molecule:
        return cumulene_conformations_per_molecule[molecule]
    raise ValueError(f"Unknown molecule: {molecule}")

molecule_to_atomic_numbers = {
    'aspirin': [6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1],
    'azobenzene': [7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'benzene': [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1],
    'ethanol': [6, 6, 8, 1, 1, 1, 1, 1, 1],
    'malonaldehyde': [6, 6, 6, 8, 8, 1, 1, 1, 1],
    'naphthalene': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1],
    'paracetamol': [6, 6, 8, 7, 6, 6, 6, 6, 8, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'salicylic_acid': [6, 6, 6, 8, 6, 6, 6, 6, 8, 8, 1, 1, 1, 1, 1, 1],
    'toluene': [6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1],
    'uracil': [6, 6, 7, 6, 7, 6, 8, 8, 1, 1, 1, 1],
    'Ac-Ala3-NHMe': [1, 1, 1, 6, 8, 6, 7, 1, 6, 1, 6, 6, 8, 1, 1, 1, 7, 1, 6, 1, 6, 6, 8, 1,
        1, 1, 7, 1, 6, 1, 6, 6, 8, 1, 1, 1, 7, 1, 6, 1, 1, 1],
    'DHA': [8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1],
    'stachyose': [6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 8, 6, 6, 6, 6, 6, 6, 8, 8, 6, 6,
        8, 6, 6, 6, 6, 8, 8, 1, 8, 1, 8, 1, 1, 8, 1, 8, 1, 8, 1, 1, 8, 1, 8, 1,
        8, 1, 8, 1, 1, 8, 1, 8, 1, 8, 1, 8, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'AT-AT': [7, 6, 7, 6, 6, 6, 7, 6, 7, 7, 1, 1, 1, 1, 1, 7, 6, 7, 6, 6, 6, 8, 8, 6,
        1, 1, 1, 1, 1, 1, 7, 6, 7, 6, 6, 6, 7, 6, 7, 7, 1, 1, 1, 1, 1, 7, 6, 7,
        6, 6, 6, 8, 8, 6, 1, 1, 1, 1, 1, 1],
    'AT-AT-CG-CG': [7, 8, 7, 6, 7, 6, 6, 6, 7, 6, 7, 1, 1, 1, 1, 1, 8, 7, 7, 6, 6, 7, 6, 6,
        1, 1, 1, 1, 1, 7, 8, 7, 6, 7, 6, 6, 6, 7, 6, 7, 1, 1, 1, 1, 1, 8, 7, 7,
        6, 6, 7, 6, 6, 1, 1, 1, 1, 1, 7, 6, 7, 6, 6, 6, 7, 6, 7, 7, 1, 1, 1, 1,
        1, 7, 6, 7, 6, 6, 6, 8, 8, 6, 1, 1, 1, 1, 1, 1, 7, 6, 7, 6, 6, 6, 7, 6,
        7, 7, 1, 1, 1, 1, 1, 7, 6, 7, 6, 6, 6, 8, 8, 6, 1, 1, 1, 1, 1, 1],
    'buckyball-catcher': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6],
    'double-walled_nanotube': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}

def molecule_to_n_atoms(molecule):
    return len(molecule_to_atomic_numbers[molecule])

ELEMENT = [
    None,
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn",
    "Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
    "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
    "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn"
]

def Z_to_symbol(Z):
    return ELEMENT[Z]
