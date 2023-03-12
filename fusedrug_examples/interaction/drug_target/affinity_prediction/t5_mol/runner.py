"""
BimodalMCA affinity predictor (see https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.1c00889)
"""

import os
from omegaconf import DictConfig, OmegaConf
import hydra
import socket
import tensorflow
# import model
# import data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import T5Tokenizer, T5Model, AutoTokenizer, T5ForConditionalGeneration

import pytoda
from pathlib import Path


CONFIGS_DIR = os.path.join(os.path.dirname(__file__), 'configs')
SELECTED_CONFIG = 'train_config.yaml'

def pytoda_ligand_tokenizer_path():
    path = Path(pytoda.__file__)
    path = os.path.join(path.parent.absolute(), 'smiles', 'metadata', 'tokenizer', 'vocab.json')
    return path


OmegaConf.register_new_resolver("pytoda_ligand_tokenizer_path", pytoda_ligand_tokenizer_path)
@hydra.main(config_path=CONFIGS_DIR, config_name=SELECTED_CONFIG)
# def run_train_and_val(cfg : DictConfig) -> None:

#     if len(cfg)==0:
#         raise Exception(f'You should provide --config-dir and --config-name  . Note - config-name should be WITHOUT the .yaml postfix')

#     SESSION_FULL_PATH = os.path.realpath(os.getcwd())
    
#     print('Hydra config:')
#     print(OmegaConf.to_yaml(cfg))
#     print('End of Hydra config.')

    
#     print(f'Running on hostname={socket.gethostname()}')
#     STOP_FILENAME = os.path.join(SESSION_FULL_PATH, 'STOP')
#     print(f'Will monitor the presence of a stop file to enable stopping a session gracefully: {STOP_FILENAME}')
#     exit_on_stopfile_callback = utils.ExitOnStopFileCallback(STOP_FILENAME)

#     OmegaConf.resolve(cfg) #to make sure that all "interpolated" values are resolved ( https://omegaconf.readthedocs.io/_/downloads/en/latest/pdf/ )
#     cfg_raw = OmegaConf.to_object(cfg)

#     lightning_data = data.AffinityDataModule(**cfg_raw['data']['lightning_data_module'])
#     lightning_model = model.AffinityPredictorModule(**cfg_raw['model'])

#     val_loss_callback = ModelCheckpoint(
#         dirpath=SESSION_FULL_PATH,
#         filename="val_loss",
#         mode="min",
#         monitor="val_loss",
#         verbose=True,
#         every_n_epochs=1,
#         save_last=True,
#         save_top_k=1,
#     )
#     val_rmse_callback = ModelCheckpoint(
#         dirpath=SESSION_FULL_PATH,
#         filename="val_rmse",
#         mode="min",
#         monitor="val_rmse",
#         verbose=True,
#         every_n_epochs=1,
#         save_top_k=1,
#     )
#     val_pearson_callback = ModelCheckpoint(
#         dirpath=SESSION_FULL_PATH,
#         filename="val_pearson",
#         mode="max",
#         monitor="val_pearson",
#         verbose=True,
#         every_n_epochs=1,
#         save_top_k=1,
#     )

#     trainer = pl.Trainer(
#         **cfg.trainer,
#         callbacks=[val_rmse_callback, val_loss_callback, val_pearson_callback, exit_on_stopfile_callback],
#     )

#     trainer.fit(lightning_model, lightning_data)


def run_t5_test(cfg : DictConfig) -> None:
    """https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Model

    Args:
        cfg (DictConfig): _description_
    """

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # training
    input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
    labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    logits = outputs.logits

    # inference
    input_ids = tokenizer(
        "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
    ).input_ids  # Batch size 1
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # studies have shown that owning a dog is good for you.
    a=1
    
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = T5Model.from_pretrained("t5-small")

    input_ids = tokenizer(
        "Studies have been shown that owning a dog is good for you", return_tensors="pt"
    ).input_ids  # Batch size 1
    decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

    # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
    # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
    decoder_input_ids = model._shift_right(decoder_input_ids)

    # forward pass
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    last_hidden_states = outputs.last_hidden_state
    
    a=1

if __name__ == '__main__':
    os.environ['TITAN_RESULTS'] = '/dccstor/fmm/users/vadimra/dev/output/TITAN/TITAN_T5/'
    os.environ['TITAN_DATA'] = '/dccstor/fmm/users/vadimra/dev/data/TITAN/08-02-2023/'
    run_t5_test()
    # run_train_and_val()









