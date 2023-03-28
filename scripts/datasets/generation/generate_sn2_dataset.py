from src.reactions.e2_sn2.e2_sn2_dataset import E2Sn2Dataset

ds = E2Sn2Dataset('sn2_dataset.csv')
ds.generate(4)
