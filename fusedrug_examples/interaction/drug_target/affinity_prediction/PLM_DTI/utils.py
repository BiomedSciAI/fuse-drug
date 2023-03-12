from pathlib import Path


def get_task_dir(task_name: str, orig_repo_name: str = ""):
    """
    Slightly modified function vs. that from the original repo,
    to allow appending the repo name, for when used as a submodule.

    Get the path to data for each benchmark data set

    :param task_name: Name of benchmark
    :type task_name: str
    """

    task_paths = {
        "biosnap": "./dataset/BIOSNAP/full_data",
        "biosnap_prot": "./dataset/BIOSNAP/unseen_protein",
        "biosnap_mol": "./dataset/BIOSNAP/unseen_drug",
        "bindingdb": "./dataset/BindingDB",
        "davis": "./dataset/DAVIS",
        "dti_dg": "./dataset/TDC",
        "dude": "./dataset/DUDe",
        "halogenase": "./dataset/EnzPred/halogenase_NaCl_binary",
        "bkace": "./dataset/EnzPred/duf_binary",
        "gt": "./dataset/EnzPred/gt_acceptors_achiral_binary",
        "esterase": "./dataset/EnzPred/esterase_binary",
        "kinase": "./dataset/EnzPred/davis_filtered",
        "phosphatase": "./dataset/EnzPred/phosphatase_chiral_binary",
    }

    return Path(orig_repo_name, task_paths[task_name.lower()]).resolve()
