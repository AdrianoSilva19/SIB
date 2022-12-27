import itertools

import numpy as np

from si.data.dataset import Dataset


class KMer:
    def __int__(self, k: int = 2, alphabet: str = "PROT"):
        """
        :param k:The k-mer length
        :type k: int
        :param alphabet: alphabet used PROT or DNA
        :type alphabet: str
        """
        self.k = k
        self.k_mers = None
        self.alphabet = alphabet.upper()
        self.set_alphabet()

    def set_alphabet(self):
        if self.alphabet == 'DNA':
            self.alphabet = 'ACTG'
        elif self.alphabet == 'PROT':
            self.alphabet = 'FLIMVSPTAY_HQNKDECWRG'
        else:
            self.alphabet = self.alphabet

    def fit(self, dataset: Dataset):
        """
        Fits the descriptor to dataset

        :return: the fitted dataset
        """
        # generate the k_mers
        self.k_mers = [''.join(k_mer) for k_mer in itertools.product(alphabet, repeat = self.k)]
        return self




    def transform(self, dataset: Dataset):
        """

        :param dataset: Dataset to be transformed
        :type dataset: pandas Dataset
        :return: the transformed Dataset
        :rtype: pandas Dataset
        """
        # calculate the k_mer composition
        sequence_k_mer_composition = [self._get_sequence_k_mer_composition(sequence) for sequence in dataset.X[:,0]]
        sequences_k_mer_composition = np.array(sequence_k_mer_composition)
        # create new dataset
        return Dataset(X=sequences_k_mer_composition, y=dataset.y, features=self.k_mers, label=dataset.labels)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        fits descriptor to Dataset
        :param dataset: the dataset to fit the descriptor to transform
        :return: transformed Dataset
        """

        return self.fit(dataset).transform(dataset)

    def _get_sequence_k_mer_composition(self, sequence: str):
        """
        Calculates k-mer composition
        :param sequence: The sequence to calculate k-mer composition
        :type sequence: str
        :return: the k-mer composition of the sequence
        :rtype: list of float
        """

        # calculate the k-mer composition
        counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i:i + self.k]
            counts[k_mer] += 1

        # normalize the counts
        return np.array([counts[k_mer] / len(sequence) for k_mer in self.k_mers])