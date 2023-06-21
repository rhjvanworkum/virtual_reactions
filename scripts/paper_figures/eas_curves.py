import matplotlib.pyplot as plt

trans_results = {
    'Baseline': {
        2214:  [0.947],
        1578:  [0.859],
        678:   [0.777],
        221:   [0.613],
    },
    'Baseline + XTB-feat': {
        2214:  [0.947],
        1578:  [0.859],
        678:   [0.777],
        221:   [0.613],
    },
    'XTB simulations': {
        2214:  [0.947],
        1578:  [0.859],
        678:   [0.777],
        221:   [0.613],
    },
    'FF simulations': {
        2214:  [0.947],
        1578:  [0.859],
        678:   [0.777],
        221:   [0.613],
    },
    'Local simulations*': {
        2214:  [0.947],
        1578:  [0.859],
        678:   [0.777],
        221:   [0.613],
    },
    'Local simulations**': {
        2214:  [0.947],
        1578:  [0.859],
        678:   [0.777],
        221:   [0.613],
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