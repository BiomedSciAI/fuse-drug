from typing import List, Optional, Union
from fuse.utils import NDict
from fuse.data import OpBase, get_sample_id
import bmfm_bench.benchmarks.interaction.drug_target_interaction.dti_binding_dataset as dbd


class DTIBindingDatasetLoader(OpBase):
    def __init__(
        self,
        pairs_tsv: str,
        ligands_tsv: str,
        targets_tsv: str,
        splits_tsv: str = None,
        use_folds: Optional[Union[List, str]] = None,
        keep_activity_labels: Optional[Union[List, str]] = None,
        cache_dir: Optional[str] = None,
        force_dummy_constant_ligand_for_debugging: bool = False,
        force_dummy_constant_target_for_debugging: bool = False,
        **kwargs: dict,
    ):
        """
        See DTIBindingDataset Doc
        """
        super().__init__()

        self.dataset = dbd.DTIBindingDataset(
            pairs_tsv=pairs_tsv,
            ligands_tsv=ligands_tsv,
            targets_tsv=targets_tsv,
            splits_tsv=splits_tsv,
            use_folds=use_folds,
            keep_activity_labels=keep_activity_labels,
            cache_dir=cache_dir,
        )

        self._force_dummy_constant_ligand_for_debugging = (
            force_dummy_constant_ligand_for_debugging
        )
        if self._force_dummy_constant_ligand_for_debugging:
            print(
                "WARNING: DEBUG MODE ACTIVATED!!!!! force_dummy_constant_ligand_for_debugging"
            )

        self._force_dummy_constant_target_for_debugging = (
            force_dummy_constant_target_for_debugging
        )
        if self._force_dummy_constant_target_for_debugging:
            print(
                "WARNING: DEBUG MODE ACTIVATED!!!!! force_dummy_constant_target_for_debugging"
            )

    def __call__(
        self,
        sample_dict: NDict,
        key_out_ligand: str = "data.input.ligand",
        key_out_target: str = "data.input.target",
        key_out_ground_truth_activity_value: str = "data.gt.activity_value",
        key_out_ground_truth_activity_label: str = "data.gt.activity_label",
    ) -> NDict:

        """ """
        sid = get_sample_id(sample_dict)
        entry = self.dataset[sid]

        ##
        sample_dict[key_out_ground_truth_activity_value] = entry[
            "ground_truth_activity_value"
        ]
        sample_dict[key_out_ground_truth_activity_label] = entry[
            "ground_truth_activity_label"
        ]

        if not self._force_dummy_constant_ligand_for_debugging:
            sample_dict[key_out_ligand] = entry["ligand_str"]
        else:
            sample_dict[
                key_out_ligand
            ] = "c1cc(NNc2cccc(-c3nn[nH]n3)c2)cc(-c2nn[nH]n2)c1"

        if not self._force_dummy_constant_target_for_debugging:
            sample_dict[key_out_target] = entry["target_str"]
        else:
            sample_dict[
                key_out_target
            ] = "MLLSKINSLAHLRAAPCNDLHATKLAPGKEKEPLESQYQVGPLLGSGGFGSVYSGIRVSDNLPVAIKHVEKDRISDWGELPNGTRVPMEVVLLKKVSSGFSGVIRLLDWFERPDSFVLILERPEPVQDLFDFITERGALQEELARSFFWQVLEAVRHCHNCGVLHRDIKDENILIDLNRGELKLIDFGSGALLKDTVYTDFDGTRVYSPPEWIRYHRYHGRSAAVWSLGILLYDMVCGDIPFEHDEEIIRGQVFFRQRVSSECQHLIRWCLALRPSDRPTFEEIQNHPWMQDVLLPQETAEIHLHSLSPGPSK"

        return sample_dict
