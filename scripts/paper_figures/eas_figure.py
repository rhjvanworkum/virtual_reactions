import matplotlib.pyplot as plt

# contains Transductive & Non-Transductive results next to each other
results = {
    # vanilla
    'Baseline': {
        2210:  [0.961, 0.961],
        1611:  [0.943, 0.943],
        756:   [0.918, 0.918],
        243:   [0.844, 0.844],
    },
    # FF
    'FF sim': {
        2210:  [0.787, 0.783],
        1611:  [0.772, 0.767],
        756:   [0.711, 0.736],
        243:   [0.708, 0.718],
    },
    # XTB
    'XTB sim feat': {
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
    # local sims (exp)
    'Local sim feat': {
        2210:  [0.944, 0.944],
        1611:  [0.974, 0.974],
        756:   [0.963, 0.963],
        243:   [0.939, 0.939],
    },
    'Local sim subset': {
        2210:  [0.981, 0.993],
        1611:  [0.989, 0.987],
        756:   [0.984, 0.986],
        243:   [0.987, 0.987],
    },
    'Local sim whole': {
        2210:  [0.956, 0.968],
        1611:  [0.962, 0.956],
        756:   [0.962, 0.962],
        243:   [0.956, 0.955],  
    },
    # local sims (xtb)
    'Local sim (XTB) feat': {
        2210:  [0.953, 0.953],
        1611:  [0.922, 0.922],
        756:   [0.891, 0.891],
        243:   [0.866, 0.866],
    },
    'Local sim (XTB) subset': {
        2210:  [0.953, 0.956],
        1611:  [0.943, 0.935],
        756:   [0.913, 0.921],
        243:   [0.895, 0.902],
    },
    'Local sim (XTB) whole': {
        2210:  [0.907, 0.892],
        1611:  [0.897, 0.899],
        756:   [0.891, 0.896],
        243:   [0.885, 0.882],  
    },
}

colors = {
    'Baseline': 'black',
    'FF sim': 'yellow',
    'XTB sim feat': 'forestgreen',
    'XTB sim': 'darkgreen',
    'Local sim feat': 'royalblue',
    'Local sim subset': 'navy',
    'Local sim whole': 'deepskyblue',
    'Local sim (XTB) feat': 'firebrick',
    'Local sim (XTB) subset': 'salmon',
    'Local sim (XTB) whole': 'lightcoral',
}
    
markers = {
    'Baseline': 'o',
    'FF sim': 'x',
    'XTB sim feat': '^',
    'XTB sim': 'x',
    'Local sim feat': '^',
    'Local sim subset': 'x',
    'Local sim whole': 'x',
    'Local sim (XTB) feat': '^',
    'Local sim (XTB) subset': 'x',
    'Local sim (XTB) whole': 'x',
}

if __name__ == "__main__":
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 8), sharex=True, sharey=True)
    titles = ['OOD test AUROC trans', 'OOD test AUROC non-trans']

    for idx in range(2):
        for key, values in results.items():
            ax[idx].plot(
                [k for k in values.keys()], 
                [v[idx] for v in values.values()], 
                color=colors[key],
                label=key,
                marker=markers[key]
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