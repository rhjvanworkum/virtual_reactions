import numpy as np

import matplotlib.pyplot as plt

results = {
    'Baseline': {
        0:  [9.76e-3, 2.24e-2],
        10: [4.90e-3, 1.42e-2],
        20: [2.88e-3, 3.20e-3],
        30: [1.41e-3, 3.97e-3]
    },
    'DFT augmented': {
        0:  [9.76e-3, 2.24e-2],
        10: [5.85e-3, 6.94e-3],
        20: [2.23e-3, 6.00e-3],
        30: [1.71e-3, 1.92e-3]
    },
    'Surrogate augmented (10%)': {
        0:  [9.76e-3, 2.24e-2],
        10: [3.97e-3, 5.97e-3],
        20: [5.69e-3, 5.76e-3],
        30: [2.10e-3, 1.97e-3]
    },
    'Surrogate augmented (2%)': {
        0:  [9.76e-3, 2.24e-2],
        10: [3.61e-3, 6.82e-2],
        20: [3.64e-3, 6.51e-3],
        30: [4.44e-3, 4.44e-3]
    },
}


if __name__ == "__main__":
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(9, 4))
    titles = ['IID test MAE', 'OOD test MAE']


    lines = []
    for idx, title in enumerate(titles):
        for key, values in results.items():
            lines.append(ax[idx].plot([k for k in values.keys()], [v[idx] for v in values.values()], label=key))
        
        ax[idx].set_yscale('log')
        # axis.set_xlim([100, 6000])
        ax[idx].set_title(title, fontsize=16)

    fig.text(
        x=0.35, 
        y=0.04, 
        s="% of total ani-1cxx data added",
        fontsize=14
    )

    fig.legend(
        handles=[l[0] for l in lines[:4]],
        loc='right', 
        bbox_to_anchor=(1.0, 0.72)
    )

    plt.subplots_adjust(
        bottom=0.17,
    )

    plt.savefig('eas_curves.png')
    # plt.subplot_tool()
    plt.show()
