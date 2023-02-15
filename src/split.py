


from typing import List


class Split:

    def __init__(
        self,
    ) -> None:
        pass



class HeteroCycleSplit:

    excluded_hetero_cycles = [
        'c1cc1S'
    ]

    def __init__(
        self,
        n_train_samples: float = 0.9,
        n_val_samples: float = 0.05,
        n_test_samples: float = 0.0,
        transductive: bool = False,
    ) -> None:
        
        self.n_train_samples = n_train_samples
        self.n_val_samples = n_val_samples
        self.n_test_samples = n_test_samples
        self.transductive = transductive

    def get_test_set(
        self,
        substrate_smiles: List[str]
    ):
        test_idxs = []
        for idx, smiles in substrate_smiles:
            for heterocycle in self.excluded_hetero_cycles:
                if len(smiles.SubstructureMatch(heterocycle)) > 0:
                    test_idxs.append(idx)
        return test_idxs
       
    def generate(
        self,
    ):
        # if transductive we incorporate also 
        # virtual reactions on the test set

        # if not transductive we don't incorporate virtaul reaction
        # on the test set
