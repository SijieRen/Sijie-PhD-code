"""
FOLIO: Natural Language Reasoning with First-Order Logic
https://arxiv.org/pdf/2209.00840.pdf
"""
from eval.base import OWAFOLTask
from eval.tasks.utils import evaluate, convert_to_nltk_rep

_CITATION = """
@article{han2022folio,
  title={Folio: Natural language reasoning with first-order logic},
  author={Han, Simeng and Schoelkopf, Hailey and Zhao, Yilun and Qi, Zhenting and Riddell, Martin and Benson, Luke and Sun, Lucy and Zubova, Ekaterina and Qiao, Yujie and Burtell, Matthew and others},
  journal={arXiv preprint arXiv:2209.00840},
  year={2022}
}
"""


def create_all_tasks():
    def create_task(mode, n):
        class FOLIO(FOLIOBase):
            def __init__(self):
                super().__init__(mode, n)

        return FOLIO

    return {
        f"folio-{mode}-{n}shot": create_task(mode, n)
        # for mode in ["baseline", "scratchpad", "neurosymbolic", "cot", "inf_lr"]
        for mode in ["inf_lr", "neurosymbolic"] 
        # for n in [1, 2, 4, 8, 16]
        for n in [1, 2, 4]
    }


class FOLIOBase(OWAFOLTask):
    # import os
    # print("FOLIO: ", os.getcwd())
    # DATASET_PATH = "benlipkin/folio"
    DATASET_PATH = "../../../data/folio/data/v0.0/folio-train.jsonl"
    DATASET_NAME = None

    def __init__(self, mode, n, seed=7):
        super().__init__(mode, n)
        # process validation dataset
        print("FOLIOBase self.dataset_test: $$$$$$$$$$$$$$$$$$",self.dataset_test)
        self._dataset = self.reformat_fol_samples(self.dataset_test["train"]).shuffle(seed)
        # print("self._dataset self._dataset self._dataset self._dataset self._dataset",self._dataset)
        self._test = self._dataset.select(range(0, len(self._dataset)))
        # print("self._test ************** self._test",self._test)

    def reformat_fol_samples(self, dataset):
        def reformat_fol_sample(sample):
            # print("FOLIOBase - reformat_fol_sample : sample111111111",sample)

            sample["premises-FOL"] = [
                convert_to_nltk_rep(premise) for premise in sample["premises-FOL"]
            ]
            # print("FOLIOBase - reformat_fol_sample : sample2222222222",sample)
            # print("sample[premises-FOL] ***************8",sample["premises-FOL"])
            sample["conclusion-FOL"] = convert_to_nltk_rep(sample["conclusion-FOL"])
            # print("FOLIOBase - reformat_fol_sample : sample33333333333",sample)

            # print("sample[conclusion-FOL *****************",sample["conclusion-FOL"])
            try:
                assert len(sample["premises"]) == len(sample["premises-FOL"])
                label = evaluate(sample["premises-FOL"], sample["conclusion-FOL"])
                assert sample["label"] == label
            except Exception as e:
                # print(f"Error in parsing FOL: {e}")
                # print(sample)
                sample["label"] = self.ERROR_TOKEN
            return sample
        # print("self.ERROR_TOKEN ^^^^^^^^^^^^^^^^^^^____________", self.ERROR_TOKEN)
        # print()
        return dataset.map(reformat_fol_sample)
        # return dataset.map(reformat_fol_sample).filter(
        #     lambda x: x["label"] != self.ERROR_TOKEN
        # )