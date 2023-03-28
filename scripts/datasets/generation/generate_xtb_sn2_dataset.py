from src.reactions.e2_sn2.e2_sn2_dataset import XtbSimulatedE2Sn2Dataset, DFTSimulatedE2Sn2Dataset
from src.reactions.e2_sn2.e2_sn2_dataset import E2Sn2Dataset

source_ds = E2Sn2Dataset('sn2_dataset.csv')

# ds = XtbSimulatedE2Sn2Dataset('xtb_simulated_sn2.csv')
ds = DFTSimulatedE2Sn2Dataset('dft_simulated_sn2.csv')
ds.generate(source_ds, 2)
