import matplotlib.pyplot as plt

results = {
    'Baseline': {
        2225:  [0.947],
        1527:  [0.859],
        600:   [0.777],
        200:   [0.613],
    },
    'Simulations (trans)': {
        2225:  [0.963],
        1527:  [0.881],
        600:   [0.885],
        200:   [0.862],
    },
    'Simulations (non-trans)': {
        2225:  [0.946],
        1527:  [0.861],
        600:   [0.836],
        200:   [0.659],
    },
}


if __name__ == "__main__":

    plt.title('OOD test AUROC')

    for key, values in results.items():
        plt.plot([k for k in values.keys()], [v[0] for v in values.values()], label=key)

    plt.gca().invert_xaxis()
    plt.yscale('log')

    plt.ylabel('AUROC')
    plt.xlabel('Number of training data points')

    plt.legend()
    plt.savefig('da_learning_curves.png')
    plt.show()