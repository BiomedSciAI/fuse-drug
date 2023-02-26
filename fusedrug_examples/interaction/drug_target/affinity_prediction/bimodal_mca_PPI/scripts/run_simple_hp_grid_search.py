import sys
from sklearn.model_selection import ParameterGrid
from typing import Union,List
from zed_dti.fully_supervised.dti_classifier.scripts.session_runner_cc import run
from zed_dti.fully_supervised.dti_classifier.utils import get_valid_filename
from pprint import pprint
from time import sleep
from hydra import compose, initialize
from omegaconf import OmegaConf
from zed_dti.fully_supervised.dti_classifier.utils import hydra_resolvers
for name,func in hydra_resolvers.items():
    OmegaConf.register_new_resolver(name, func)
#TODO: copy the contents of this file into running files inside the session group dir (run_simple_hp_grid_search_20.py etc.)
def run_grid_search(desc:Union[dict,List[dict]], dry_run:bool=False, undesired_subdicts=List[dict], **session_runner_kwargs):
    """
    Runs a grid search on CCC using the search description as defined in desc.
    Args:
        desc: see sklearn.model_selection.ParameterGrid docstring for details.
            basic example:
            desc={'lr':[1e-4,4e-5,1e-5], 'grad_clip':[None, 0.3, 3.0]}
            would result in running:
            [{'grad_clip': None, 'lr': 0.0001},
            {'grad_clip': None, 'lr': 4e-05},
            {'grad_clip': None, 'lr': 1e-05},
            {'grad_clip': 0.3, 'lr': 0.0001},
            {'grad_clip': 0.3, 'lr': 4e-05},
            {'grad_clip': 0.3, 'lr': 1e-05},
            {'grad_clip': 3.0, 'lr': 0.0001},
            {'grad_clip': 3.0, 'lr': 4e-05},
            {'grad_clip': 3.0, 'lr': 1e-05}]

            instead of a dict, desc can be a list of dictionaries which is useful in the case of 
            combinations of params that don't make sense. (see sklearn.model_selection.ParameterGrid for more details about this as well)

    """

    print('*******************')
    print(' run_grid_search()')
    print('*******************')

    print('input desc:')
    pprint(desc)

    print('input undesired_subdicts:')
    pprint(undesired_subdicts)

    print('input session_runner_kwargs:')
    pprint(session_runner_kwargs)
    
    all_configs = list(ParameterGrid(desc))
    print(f'the input desc contains a total of {len(all_configs)} runs')

    all_configs = [x for x in all_configs if not check_if_any_match(x, undesired_subdicts)]
    print(f'after applying undesired_subdicts (by removing) {len(all_configs)} remained')

    ans = input('Do you want to continue? type exactly "yes" if you do: ')
    if ans != 'yes':
        print('user cancelled.')
        sys.exit(1)

    for idx, conf in enumerate(all_configs):        
        print(f'\nconf {idx}:')
        pprint(conf)
        
        # ###convert values into the expected way in yaml/hydra config

        # #None -> null
        # conf = {k:'null' if d is None else d for (k,d) in conf.items()}
        
        # ##True -> true
        # conf = {k:'true' if d == True else d for (k,d) in conf.items()}

        # ##False -> false
        # conf = {k:'false' if d == False else d for (k,d) in conf.items()}

        #in hydra prefixing with + is override, and prefixing with ++ is change or override
        args = [f'++{k}="{d}"' for (k,d) in conf.items()]
        args = ' '.join(args)

        print('hydra extra args: ', args)

        if dry_run:
            continue

        run(run_cli_args=args,
            **session_runner_kwargs
        )

        sleep(10) #otherwise clearml goes crazy
        

def check_if_any_match(full_dict, query_dicts):
    for q in query_dicts:
        if check_if_matching(full_dict,q):
            return True
    return False

def check_if_matching(full_dict, query_dict):
    for k,d in query_dict.items():
        if k not in full_dict:
            return False
        if full_dict[k] != query_dict[k]:
            return False
    return True


if __name__=='__main__':
    initialize(config_path="../configs") #, job_name="test_app")    
    cfg = compose(config_name="train_config") # remember that we can provide overrides: overrides=["db=mysql", "db.user=me"]
    OmegaConf.resolve(cfg) 
    
    session_group_name = cfg['session_group_name']
    session_group_name = get_valid_filename(session_group_name)

    run_grid_search(
        desc=[{
                
                ########'data.lightning_data_module.force_dummy_constant_target_for_debugging' : [False, True],
                ########'data.lightning_data_module.training_ligand_augmentation_shuffle_atoms' : [False, True],
                ########'data.lightning_data_module.training_target_augmentation_flip_full' : [False, True],
                ########'data.lightning_data_module.training_target_augmentation_add_noise' : [None, 0.05, 0.1,0.2],

                ##'data.lightning_data_module.num_pseudo_msa' : ['0','1'],
                
                # 'trainer.gradient_clip_val' : ['null', '0.6'],                
                # 'trainer.accumulate_grad_batches' : ['null', 3],                
                # 'model.learning_rate' : ['1e-4', '1e-5'],
                # 'model.model_kwargs.use_target_sequence_input': ['true', 'false'],
                # 'model.model_kwargs.use_target_plm_embedding_input': ['true', 'false'],                

                # 'model.learning_rate' : ['1e-4', '1e-5'],
                # 'model.loss_fn' : ['focal', 'cross_entropy'],
                # 'model.model_kwargs.internal_model_kwargs.attn_dropout' : ['0.05', '0.2'],
                # 'model.model_kwargs.internal_model_kwargs.ff_dropout' : ['0.05', '0.2'],

                
                # 'model.model_kwargs.use_target_sequence_input': ['false'],
                # 'model.model_kwargs.use_target_plm_embedding_input': ['true'],                
                # 'model.learning_rate' : ['1e-4', '4e-5', '1e-5'],
                # 'model.loss_fn' : ['focal'],
                # 'model.model_kwargs.internal_model_kwargs.attn_dropout' : ['0.0', '0.3'],
                # 'model.model_kwargs.internal_model_kwargs.ff_dropout' : ['0.0', '0.3'],
                # # 'data.lightning_data_module.training_target_augmentation_flip_full' : ['true', 'false'],
                # # 'data.lightning_data_module.training_target_augmentation_add_noise' : ['0.15', 'null'],
                # # 'data.lightning_data_module.training_ligand_augmentation_shuffle_atoms' : ['true', 'false'],
                # # 'data.lightning_data_module.training_target_augmentation_flip_full' : ['true'],
                # # 'data.lightning_data_module.training_target_augmentation_add_noise' : ['0.15'],
                # # 'data.lightning_data_module.training_ligand_augmentation_shuffle_atoms' : ['true'],
                # 'data.lightning_data_module.training_target_augmentation_flip_full' : ['false'],
                # 'data.lightning_data_module.training_target_augmentation_add_noise' : ['null'],
                # 'data.lightning_data_module.training_ligand_augmentation_shuffle_atoms' : ['false'],
                
                'model.model_kwargs.use_target_sequence_input': ['false'],
                #'model.model_kwargs.use_target_sequence_input': ['true'],
                
                'model.model_kwargs.use_target_plm_embedding_input': ['true'],
                #'model.model_kwargs.use_target_plm_embedding_input': ['false'],
                'model.model_kwargs.plm_nodes_embedding_reduce': ['[mean, min, max]'],                
                #'model.model_kwargs.plm_nodes_embedding_reduce': ['null'],

                'model.learning_rate' : ['1e-4', '1e-5'],
                'model.loss_fn' : ['focal'],
                
                'model.model_kwargs.internal_model_kwargs.attn_dropout' : ['0.0'],
                'model.model_kwargs.internal_model_kwargs.ff_dropout' : ['0.0'],

                # 'model.model_kwargs.internal_model_kwargs.attn_dropout' : ['0.3'],
                # 'model.model_kwargs.internal_model_kwargs.ff_dropout' : ['0.3'],

                #'model.model_kwargs.internal_model_kwargs.attn_dropout' : ['0.5'],
                #'model.model_kwargs.internal_model_kwargs.ff_dropout' : ['0.5'],
                
                'data.lightning_data_module.training_target_augmentation_flip_full' : ['false'],
                'data.lightning_data_module.training_target_augmentation_add_noise' : ['null'],
                'data.lightning_data_module.training_ligand_augmentation_shuffle_atoms' : ['false'],

                # 'data.lightning_data_module.training_target_augmentation_flip_full' : ['true'],
                # 'data.lightning_data_module.training_target_augmentation_add_noise' : ['0.1'],
                # 'data.lightning_data_module.training_ligand_augmentation_shuffle_atoms' : ['true'],
                                
            }            
        ],
        dry_run=False,
        undesired_subdicts=[
            {'model.model_kwargs.use_target_sequence_input':'false',
            'model.model_kwargs.use_target_plm_embedding_input':'false'
            },
            {
                'data.lightning_data_module.num_pseudo_msa':'2',
                'model.model_kwargs.use_target_plm_embedding_input':'false',
            }
        ],
        #rolling_jobs=24*3,
        #rolling_jobs=2,
        
        rolling_jobs=4*3,
        pyutils_run_kwargs_override=dict(
            duration='6h',
        ),


        session_group_name=session_group_name,
    )