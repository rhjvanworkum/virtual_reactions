import random
import matplotlib.pyplot as plt
import numpy as np


BASELINE = [2.11e-2, 4.02e-2]

ablation_results = {
    f'Baseline augmented (10%)': [1.16e-2, 2.00e-2],
    f'Baseline augmented (2%)': [1.47e-2, 4.089e-2]
}
colors = ['r', 'b']

results = {
    'DFT augmented': {
        0:  [2.11e-2, 4.02e-2],
        2573: [1.16e-2, 2.00e-2],
        5147: [5.30e-3, 1.03e-2],
        7721: [5.88e-3, 7.35e-3]
    },
    'Surrogate augmented (10%)': {
        0:  [2.11e-2, 4.02e-2],
        2573: [8.93e-3, 1.54e-2],
        5147: [5.30e-3, 9.28e-3],
        7721: [2.07e-3, 3.59e-3]
    },
    'Surrogate augmented (2%)': {
        0:  [2.11e-2, 4.02e-2],
        2573: [7.71e-3, 1.65e-2],
        5147: [5.49e-3, 1.34e-2],
        7721: [2.34e-3, 4.53e-3]
    },
}


if __name__ == "__main__":
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 5))

    lines = []

    """ Bottom plot """
    titles = ['IID test MAE', 'OOD test MAE']
    for idx, title in enumerate(titles):
        for key, values in results.items():
            lines.append(
                ax[idx].plot(
                    [k for k in values.keys()], 
                    [abs(v[idx] - BASELINE[idx]) / BASELINE[idx] * 100 for v in values.values()], 
                    label=key,
                    marker='x',
                )
            )

        for key, values in ablation_results.items():
            lines.append(
                ax[idx].axhline(
                    y=abs(values[idx] - BASELINE[idx]) / BASELINE[idx] * 100, 
                    label=key,
                    # color=colors[int(random.random() * 2)],
                    linestyle='--'
                )
            )

        ax[idx].set_ylim([0, 100])
        ax[idx].set_yticklabels([f'{x}%' for x in ax[idx].get_yticks()])
        ax[idx].set_title(title)

    ax[0].set_ylabel('Performance gain (%) w.r.t baseline')

    handles = [l[0] for l in lines[:3]] + [l for l in lines[3:5]]
    fig.legend(
        handles=handles,
        loc='lower right', 
        bbox_to_anchor=(1.0, 0.2)
    )

    fig.text(
        x=0.35, 
        y=0.04, 
        s='Number of added simulated data samples',
        fontsize=12
    )

    plt.subplot_tool()
    plt.tight_layout()

    plt.subplots_adjust(
        wspace=0.345,
        bottom=0.14,
    )

    # plt.savefig('ani-cxx_figure.png', dpi=800)
    plt.show()