import matplotlib.pyplot as plt
import numpy as np


ten_percent_dict = {
    "Compound 1":  [8.55e-5, 1.10e-4],
    "Compound 2":  [1.35e-4, 1.04e-4],
    "Compound 3":  [8.81e-5, 1.01e-4],
    "Compound 4":  [1.11e-4, 1.14e-4],
    "Compound 5":  [1.26e-4, 1.19e-4]
}
two_percent_dict = {
    "Compound 1": [1.33e-4, 8.84e-5],
    "Compound 2": [1.15e-4, 1.21e-4],
    "Compound 3": [3.34e-4, 2.90e-4],
    "Compound 4": [8.69e-5, 9.94e-5],
    "Compound 5": [1.35e-4, 1.13e-4]
}


if __name__ == "__main__":
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(9, 4))
    titles = ['Training MAE', 'Validation MAE']

    lines = []

    for idx, (key, values) in enumerate(ten_percent_dict.items()):
        lines.append(
            ax[0].bar(
                idx, 
                values[0], 
                label=f'{key} (10%)'
            )    
        )

    for key, values in two_percent_dict.items():
        lines.append(
            ax[1].bar(
                idx, 
                values[0], 
                label=f'{key} (2%)'
            )      
        )

    fig.legend(
        handles=[l[0] for l in lines],
        loc='right', 
        # bbox_to_anchor=(1.0, 0.72)
    )

    plt.show()