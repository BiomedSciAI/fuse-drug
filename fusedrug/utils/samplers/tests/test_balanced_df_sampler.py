import unittest
from fusedrug.utils.samplers.balanced_df_sampler import BalancedClassDataFrameSampler
import pandas as pd
import numpy as np


class TestBalancedClassDataFrameSampler(unittest.TestCase):
    def test_balanced_df_sampler(self) -> None:
        df = pd.DataFrame(
            {
                "index": ["sample" + str(i) for i in range(1000)],
                "label": np.random.choice(["1", "2"], 1000, p=[0.8, 0.2]).tolist(),
            }
        )
        df.set_index("index", drop=True, inplace=True)
        sampler = BalancedClassDataFrameSampler(
            df=df,
            label_column_name="label",
            classes=["1", "2"],
            counts=[5, 5],
            shuffle=False,
            total_minibatches=1000,
        )
        # TODO: come up with better test for shuffle=True

        indices_1 = np.flatnonzero(df["label"] == "1")
        indices_2 = np.flatnonzero(df["label"] == "2")

        minibatches_needed_to_see_1 = int(np.ceil(indices_1.shape[0] / 5))
        minibatches_needed_to_see_2 = int(np.ceil(indices_2.shape[0] / 5))

        seen_1 = {}
        seen_2 = {}
        seen_all_1s_after = []
        seen_all_2s_after = []
        for i, mb in enumerate(sampler):
            # check frequency per batch
            df_1 = df.loc[mb][df.loc[mb]["label"] == "1"]
            df_2 = df.loc[mb][df.loc[mb]["label"] == "2"]
            self.assertEqual((df.loc[mb].label == "1").sum(), 5)
            self.assertEqual((df.loc[mb].label == "2").sum(), 5)
            for k in df_1.index:
                if k not in seen_1:
                    seen_1[k] = 1
                else:
                    seen_all_1s_after.append(i + 1)
                    seen_1 = {}

            for k in df_2.index:
                if k not in seen_2:
                    seen_2[k] = 1
                else:
                    seen_all_2s_after.append(i + 1)
                    seen_2 = {}

        self.assertAlmostEqual(seen_all_1s_after[0], minibatches_needed_to_see_1, delta=1)
        self.assertAlmostEqual(seen_all_2s_after[0], minibatches_needed_to_see_2, delta=1)

        self.assertAlmostEqual(
            np.mean(np.diff(np.array(seen_all_1s_after))), seen_all_1s_after[0], delta=1
        )
        self.assertAlmostEqual(np.std(np.diff(np.array(seen_all_1s_after))), 0, delta=1)
        self.assertAlmostEqual(
            np.mean(np.diff(np.array(seen_all_2s_after))), seen_all_2s_after[0], delta=1
        )
        self.assertAlmostEqual(np.std(np.diff(np.array(seen_all_2s_after))), 0, delta=1)


if __name__ == "__main__":
    unittest.main()
