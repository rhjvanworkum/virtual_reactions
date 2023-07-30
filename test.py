
a = [1, 2, 3, 1]

print(a.index(1))



# import seaborn as sn
# import numpy as np
# import matplotlib.pyplot as plt

# data = [[ 5.43217366, 6.8014695 ,  6.8666586 ,  7.822923  , 5.83868584],
#         [ 0.16579917, 3.17806352, 2.28671819 , 2.45360363 , 1.29563846],
#         [ 4.93010134, 9.55179092, 10.68214974, 10.97396472, 2.9336707 ],
#         [ 0.48410873, 1.68368126, 2.10387384 , 3.74912867 , 1.85132639],
#         [ 1.01775229, 0.51558804, 1.30241114 , 1.27663514 , 1.08277127]]
# data = np.array(data)
# for i in range(5):
#     data[i, :] = (data[i, :] - np.min(data[i, :])) / (np.max(data[i, :]) - np.min(data[i, :]))

# sn.heatmap(data, annot=True, fmt='.2f', cmap='Blues')
# plt.show()





# import numpy as np

# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_dft_0.npz')
# base = len(split['train_idx']) + len(split['val_idx'])

# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_dft_10.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)
# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_dft_20.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)
# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_dft_30.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)

# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_10.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)
# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_20.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)
# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_30.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)

# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_10.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)
# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_20.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)
# split = np.load('./data/ani-cxx/experiment_2/splits/cc_5_mol_sim_small_30.npz')
# print(len(split['train_idx']) + len(split['val_idx']) - base)