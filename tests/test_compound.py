from src.compound import Compound

def test_conformer_generation():
    substrate = 'CC(=O)ON'
    compound = Compound.from_smiles(substrate)
    compound.generate_conformers()
    compound.optimize_conformers()

    assert len(compound.conformers) > 3


if __name__ == "__main__":
    test_conformer_generation()