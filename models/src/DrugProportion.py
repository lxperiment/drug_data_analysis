import numpy as np
import pandas as pd

class DrugProportion:
    def __init__(self, date, drug_proportion):
        self._date = date
        self._drug_proportion = pd.DataFrame(drug_proportion)
        self._drug_proportion.columns = ['Drug', 'Proportion']

    def get_drug_proportion(self):
        return self._drug_proportion

    def get_date(self):
        return self._date

    def set_drug_proportion(self, drug_proportion):
        self._drug_proportion = pd.DataFrame(drug_proportion)
        self._drug_proportion.columns = ['Drug', 'Proportion']

    def set_date(self, date):
        self._date = date

    def add_drug_proportion(self, drug, proportion):
        self._drug_proportion = self._drug_proportion.append({'Drug': drug, 'Proportion': proportion}, ignore_index=True)


    def sort_drug_proportion(self, ascending=False):
        if ascending:
            self._drug_proportion = self._drug_proportion.sort_values(by='Proportion', ascending=True)
        else:
            self._drug_proportion = self._drug_proportion.sort_values(by='Proportion', ascending=False)

    def sum_drug_proportion(self):
        return np.sum(self._drug_proportion['Proportion'])

    def drug_to_list(self):
        return self._drug_proportion.Drug.tolist()

    def drug_to_set(self):
        return set(self._drug_proportion.Drug.tolist())

    def __str__(self):
        return f"Date: {self._date}\n{self._drug_proportion}"
    def __repr__(self):
        return f"Date: {self._date}\n{self._drug_proportion}"
    def __eq__(self, other):
        return self._date == other._date and self._drug_proportion.equals(other._drug_proportion)
    def __ne__(self, other):
        return not self.__eq__(other)
    def __hash__(self):
        return hash((self._date, tuple(self._drug_proportion.values.tolist())))
    def __len__(self):
        return len(self._drug_proportion)
    def __getitem__(self, key):
        return self._drug_proportion[key]
    def __setitem__(self, key, value):
        self._drug_proportion[key] = value
    def __delitem__(self, key):
        del self._drug_propotion[key]
    def __iter__(self):
        return iter(self._drug_proportion)
    def __contains__(self, item):
        return item in self._drug_proportion.values.tolist()
    def __add__(self, other):
        return self._drug_proportion.append(other._drug_proportion)