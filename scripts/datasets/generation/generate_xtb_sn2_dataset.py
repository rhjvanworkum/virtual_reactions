from src.reactions.e2_sn2.e2_sn2_dataset import *
from src.reactions.e2_sn2.e2_sn2_dataset import E2Sn2Dataset

source_ds = E2Sn2Dataset('sn2_dataset.csv')

# ds = XtbSimulatedE2Sn2Dataset('xtb_simulated_sn2.csv')
# ds.generate(source_ds, 4)

# ds = PyscfSimulatedE2Sn2Dataset('pyscf_b3lyp_6-311g_simulated_sn2.csv')
# ds.generate(source_ds, 10)

ds = FFSimulatedE2Sn2Dataset('ff_4_simulated_sn2.csv', n_simulations=4)
ds.generate(source_ds, 10)


# ds = DFTSimulatedE2Sn2Dataset('dft_simulated_sn2.csv')
