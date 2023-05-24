from typing import List
from src.data.splits import Split

class VirtualReactionSplit(Split):

    def __init__(
        self,
        transductive: bool = True
    ) -> None:
        self.transductive = transductive

    def _get_ood_test_set_uids(
        self
    ) -> List[int]:
        """
        Returns the OOD test set, experimental data that is consider out-of-distribution
        """
        raise NotImplementedError

    def _get_iid_test_set_uids(
        self
    ) -> List[int]:
        """
        Returns the IID test set, experimental data that is consider in-distribution
        """
        raise NotImplementedError
    
    def _get_virtual_test_set_uids(
            self
        ) -> List[int]:
        """
        Returns the virtual test set, just some simulated test set
        """
        raise NotImplementedError
    
    def _get_excluded_set_uids(
            self
        ) -> List[int]:
        """
        Returns the excluded set, certain simulated reactions might be excluded from training at all,
        due to the split being non-transductive
        """
        raise NotImplementedError
