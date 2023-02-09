# virtual_reactions

-> also get (decent) correlation on E2

-> try BO on E2/Sn2
    -> take 90 sample subset & see if this also works on whole set

plan:
1. Create Dataset object that loads in CSV files containing all reactions with 0's and 1's
2. Create SimulatedDataset object that takes source dataset & augments with simulations
    - code for querying Chembl compounds on similarity as well
3. Code split as well
4. Add function to dataset to take split and save csv path's somewhere
    - This function should (a) convert to Reaction SMILES & (b) add Dummy atom indicating Virtual reaction idx
5. Write automated ChemProp training / evaluation script