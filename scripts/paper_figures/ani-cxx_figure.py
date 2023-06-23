import matplotlib.pyplot as plt
import numpy as np


ten_percent_dict = {
    "Mol 1":  [8.55e-5, 1.10e-4],
    "Mol 2":  [1.35e-4, 1.04e-4],
    "Mol 3":  [8.81e-5, 1.01e-4],
    "Mol 4":  [1.11e-4, 1.14e-4],
    "Mol 5":  [1.26e-4, 1.19e-4]
}
two_percent_dict = {
    "Mol 1": [1.33e-4, 8.84e-5],
    "Mol 2": [1.15e-4, 1.21e-4],
    "Mol 3": [3.34e-4, 2.90e-4],
    "Mol 4": [8.69e-5, 9.94e-5],
    "Mol 5": [1.35e-4, 1.13e-4]
}

BASELINE = [9.76e-3, 2.24e-2]

results = {
    # 'Baseline': {
    #     0:  [9.76e-3, 2.24e-2],
    #     10: [4.90e-3, 1.42e-2],
    #     20: [2.88e-3, 3.20e-3],
    #     30: [1.41e-3, 3.97e-3]
    # },
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
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))

    lines = []

    """ Top plot 1"""
    for idx, v in enumerate(ten_percent_dict.values()):
        lines.append(ax[0, 0].bar([2 * idx], [v[0]], label="Training MAE", color='b'))
        lines.append(ax[0, 0].bar([2 * idx + 1], [v[1]], label="Validation MAE", color='orange'))
    ax[0, 0].set_xticks([0.5, 2.5, 4.5, 6.5, 8.5], ten_percent_dict.keys())
    ax[0, 0].set_title('MAE per mol on 10% dataset')
    ax[0, 0].set_ylim([0, 0.0004])

    """ Top plot 2"""
    for idx, v in enumerate(two_percent_dict.values()):
        ax[0, 1].bar([2 * idx], [v[0]], label="Training MAE", color='b')
        ax[0, 1].bar([2 * idx + 1], [v[1]], label="Validation MAE", color='orange')
    ax[0, 1].set_xticks([0.5, 2.5, 4.5, 6.5, 8.5], ten_percent_dict.keys())
    ax[0, 1].set_title('MAE per mol on 2% dataset')
    ax[0, 1].set_ylim([0, 0.0004])


    """ Bottom plot """
    titles = ['IID test MAE', 'OOD test MAE']
    for idx, title in enumerate(titles):
        for key, values in results.items():
            lines.append(
                ax[1, idx].plot(
                    [k for k in values.keys()], 
                    [v[idx] for v in values.values()], 
                    label=key,
                    marker='x',
                )
            )
        ax[1, idx].set_yscale('log')
        ax[1, idx].set_ylim([1e-3, 8e-2])
        ax[1, idx].set_title(title)
        lines.append(
            ax[1, idx].plot(
                [k for k in values.keys()], 
                [BASELINE[idx] for v in values.values()],
                label='Baseline',
                ls='--',
            )
        )

    handles = [l for l in lines[:2]] + [l[0] for l in lines[14:]]
    fig.legend(
        handles=handles,
        loc='upper right', 
        bbox_to_anchor=(1.0, 0.9)
    )

    plt.subplot_tool()

    plt.subplots_adjust(
        wspace=0.36,
        top=0.94,
        bottom=0.06,
        right=0.85
    )

    plt.show()