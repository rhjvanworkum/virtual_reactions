import matplotlib.pyplot as plt

da_results = {
    'Baseline': {
        2148:  0.947,
        1448:  0.859,
        548:   0.777,
        148:   0.613,
    },
    'Sim (features)': {
        2148:  0.897,
        1448:  0.866,
        548:   0.817,
        148:   0.788,
    },
    'Sim (trans)': {
        2148:  0.963,
        1448:  0.881,
        548:   0.885,
        148:   0.862,
    },
    'Sim (non-trans)': {
        2148:  0.946,
        1448:  0.861,
        548:   0.836,
        148:   0.659,
    },
}

snar_results = {
    'Baseline': {
        3713:  0.976,
        2573:  0.952,
        1243:  0.938,
        293:   0.805,
        198:   0.709,
    },
    'Sim (features)': {
        3713:  0.960,
        2573:  0.965,
        1243:  0.907,
        293:   0.809,
        198:   0.804,
    },
    'Sim (trans)': {
        3713:  0.935,
        2573:  0.921,
        1243:  0.916,
        293:   0.890,
        198:   0.883,
    },
    'Sim (non-trans)': {
        3713:  0.930,
        2573:  0.932,
        1243:  0.911,
        293:   0.893,
        198:   0.884,
    },
}

colors = {
    'Baseline': 'black',
    'Sim (features)': 'royalblue',
    'Sim (trans)': 'navy',
    'Sim (non-trans)': 'deepskyblue',
}

# snar_colors =  {
#     'Baseline': 'black',
#     'Sim (features)': 'firebrick',
#     'Sim (trans)': 'salmon',
#     'Sim (non-trans)': 'lightcoral',
# }


linestyles = {
    'Baseline': 'solid',
    'Sim (features)': '--',
    'Sim (trans)': 'solid',
    'Sim (non-trans)': 'solid',
}


if __name__ == "__main__":
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=False, sharey=True)
    titles = ['DA OOD test auroc', 'SnAr OOD test auroc']
    lines = []

    ax[0].invert_xaxis()
    ax[1].invert_xaxis()

    for i, (results, colors) in enumerate(zip([da_results, snar_results], [colors, colors])):
        for key, value in results.items():
            lines.append(
                ax[i].plot(
                    [k for k in value.keys()], 
                    [v for v in value.values()], 
                    label=key, 
                    color=colors[key], 
                    marker='x',
                    linestyle=linestyles[key]
                )
            )
        ax[i].set_title(titles[i])

    fig.text(
        x=0.45, 
        y=0.015, 
        s='Training set size',
        fontsize=12
    )
    fig.text(
        x=0.035, 
        y=0.45, 
        rotation='vertical',
        s='AUROC',
        fontsize=12
    )

    plt.legend(
        handles=[lines[idx][0] for idx in [0, 1, 2, 3]],
        loc='lower left',
        # bbox_to_anchor=(1.25, 0.5),
    )
    # plt.subplots_adjust(
    #     right=0.80,
    #     left=0.1,
    #     bottom=0.1,
    #     top=0.9
    # )

    # plt.subplot_tool()
    plt.show()