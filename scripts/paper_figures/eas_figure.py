import matplotlib.pyplot as plt

# contains Transductive & Non-Transductive results next to each other
results = {
    'Baseline': {
        2210:  [0.961, 0.961],
        1611:  [0.943, 0.943],
        756:   [0.918, 0.918],
        243:   [0.844, 0.844],
    },
    'Baseline + XTB-feat': {
        2210:  [0.962, 0.962],
        1611:  [0.953, 0.953],
        756:   [0.926, 0.926],
        243:   [0.882, 0.882],
    },
    'XTB sim': {
        2210:  [0.975, 0.978],
        1611:  [0.966, 0.954],
        756:   [0.956, 0.959],
        243:   [0.952, 0.948],
    },
    # 'FF sim': {
    #     2210:  [0.947],
    #     1611:  [0.859],
    #     756:   [0.777],
    #     243:   [0.613],
    # },
    'Local sim*': {
        2210:  [0.981, 0.993],
        1611:  [0.989, 0.987],
        756:   [0.984, 0.986],
        243:   [0.987, 0.987],
    },
    # 'Local sim**': {
    #     2210:  [0.947],
    #     1611:  [0.859],
    #     756:   [0.777],
    #     243:   [0.613],
    # },
    'Local XTB sim*': {
        2210:  [0.953, 0.956],
        1611:  [0.943, 0.935],
        756:   [0.913, 0.921],
        243:   [0.895, 0.902],
    },
}


if __name__ == "__main__":
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True)
    titles = ['OOD test AUROC trans', 'OOD test AUROC non-trans']

    for idx in range(2):
        for key, values in results.items():
            ax[idx].plot(
                [k for k in values.keys()], 
                [v[idx] for v in values.values()], 
                label=key,
                marker='x'
            )
        ax[idx].set_title(titles[idx])

    fig.text(
        x=0.45, 
        y=0.04, 
        s='Training set size',
        fontsize=12
    )

    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()