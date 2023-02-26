import os
import click
import optuna
from omegaconf import OmegaConf
from glob import glob
from typing import Optional
from pathlib import Path
from fuse.utils import NDict
from collections import defaultdict
from zed_dti.fully_supervised.dti_classifier.utils import get_local_timestamp
from tqdm import tqdm
#/dccstor/fmm/users/yoels/dev/sessions/dti/dti@debug_optuna_cycle_10@yoels/

#python /dccstor/fmm/users/yoels/dev/repos/main/zed_dti/zed_dti/fully_supervised/dti_classifier/scripts/update_optuna.py /dccstor/fmm/users/yoels/dev/sessions/dti/dti@optuna_cold_ligand_split_cycle_10_averaged_embeddings@yoels/ $OPTUNA_DB_URL val_rocauc_epoch maximize 

#TODO: go over all config files under the session group dir to collect all possible HP values from configs,
#as Optuna doesn't allow modifying the parameter possible values after the first report

@click.command()
@click.argument('sessions-group_dir')
@click.argument('optuna-storage')
@click.argument('metric')
@click.argument('direction')
@click.option('--study-name', default=None)
@click.option('--drop-identical-hp', is_flag=True, show_default=True, default=True)
@click.option('--only-completed', is_flag=True, show_default=True, default=False)
@click.option('--ignore-hp-params', default='paths.session_dir@load_from_checkpoint', help='a @ separated list of hp params to ignore. Useful, for example, to avoid treating session_directory as a separate HP')
def main(sessions_group_dir:str, optuna_storage:str, metric:str, direction:str, study_name:Optional[str]=None, drop_identical_hp:bool=True, only_completed:bool=False, ignore_hp_params:str='paths.session_dir@load_from_checkpoint'):
    ignore_hp_params_list = ignore_hp_params.split('@')

    ## first path, to get all possible values (needed for Optuna)
    possible_values = defaultdict(set)
    print('only_completed=', only_completed)
    identify_file_pattern = 'completed' if only_completed else 'monitor@'+metric
    print('first pass, to understand the set of possible HP values for Optuna (as it requires...)')
    all_found = glob(os.path.join(sessions_group_dir,f'./**/{identify_file_pattern}'), recursive=True)
    
    for found in tqdm(all_found, total=len(all_found)):
        found = os.path.realpath(found)

        if study_name is None:
            #deduce it from the path
            splt = Path(found).parts
            i = splt.index('monitors')
            assert i>0
            study_name = splt[i-4]
            print('deduced study_name from the first found entry: ', study_name)

        all_hydra_runs = glob(os.path.join(found[:found.find('rank_0')],'./run_*/resolved_config.yaml'), recursive=True)
        if len(all_hydra_runs)==0:
            print('warning: could not find hydra runs for:', found)
            continue

        config = OmegaConf.load(all_hydra_runs[0]) #only using the first one, assuming no change of config between hydra runs of the *same session*
        config_raw = OmegaConf.to_object(config)
        ndict_conf = NDict(config_raw)

        for k in ndict_conf.keypaths():
            if k in ignore_hp_params_list:
                continue
            possible_values[k].add(str(ndict_conf[k]))

    possible_values = {k:list(d) for (k,d) in possible_values.items()}
    print('total keys=',len(possible_values))
    if drop_identical_hp:
        possible_values = {k:d for (k,d)in possible_values.items() if len(d)>1}
        print(f'after dropping identical hyper param keys, got {len(possible_values)} keys')
        print(f'kept hyper params:')
        for k in possible_values.keys():
            print(k)
        print('------------')

    
    print('creating study object',study_name)
    try:
        optuna_study = optuna.create_study(direction=direction,
            study_name=study_name,
            storage=optuna_storage, 
            load_if_exists=False)
    except optuna.exceptions.DuplicatedStudyError:
        print(f'already found optuna study {study_name}, creating a new one instead with a different name')
        study_name+='@'+get_local_timestamp('Israel')
        optuna_study = optuna.create_study(direction=direction,
            study_name=study_name,
            storage=optuna_storage, 
            load_if_exists=False)
    print(f'done creating study object for study {study_name}')
            
    updated_user_attrs = False
    
    print('second pass, updating Optuna study with the metric values, and the HPs')
    for found in tqdm(all_found, total=len(all_found)):
        found = os.path.realpath(found)
        print(found)

        #/dccstor/fmm/users/yoels/dev/sessions/dti/dti@debug_optuna_cycle_13@yoels/549/DEPLOYED_dti_classifier/rank_0/monitors/completed
        all_hydra_runs = glob(os.path.join(found[:found.find('rank_0')],'./run_*/resolved_config.yaml'), recursive=True)
        if len(all_hydra_runs)==0:
            continue

        config = OmegaConf.load(all_hydra_runs[0]) #only using the first one, assuming no change of config between hydra runs of the *same session*
        config_raw = OmegaConf.to_object(config)
        ndict_conf = NDict(config_raw)

        if not updated_user_attrs:
            optuna_study.set_user_attr('_free_text_description', ndict_conf['_free_text_description'])
            updated_user_attrs = True
        
        #only go over the key values in possible_values
        #(which was potentially filtered to keep only non-identical values accross the HP search)
        report_hp = {k:str(ndict_conf[k]) for k in possible_values.keys()}        
        optuna_study.enqueue_trial(report_hp)
        
        optuna_trial = optuna_study.ask() # `trial` is a `Trial` and not a `FrozenTrial`.
        got_hp = {}
        for k,options_list in possible_values.items():
            #suggested_val = optuna_trial.suggest_categorical(k, sorted([str(_) for _ in options_list])) #sorting to make it consistent (otherwise optuna throws an exception that it changed)
            suggested_val = optuna_trial.suggest_categorical(k, sorted(options_list)) #sorting to make it consistent (otherwise optuna throws an exception that it changed)
            assert str(suggested_val) == str(report_hp[k])
        
        optuna_trial.set_user_attr('paths.session_dir', ndict_conf['paths.session_dir'])
        optuna_trial.set_user_attr('_free_text_description', ndict_conf['_free_text_description'])

        metric_file = os.path.join(os.path.dirname(found), 'monitor@'+metric)
        last_val = None
        with open(metric_file, 'r') as f:
            for l in f.readlines():
                iteration, val = l.split('@')
                iteration = int(iteration)
                val = float(val)
                #print(f'found iteration={iteration} value={val}')
                optuna_trial.report(val, iteration)
                last_val = val

        #currently only supporting taking last value and not best one 
        #this is probably good in the sense of reducing overfitting
        optuna_study.tell(optuna_trial, last_val)


    print('done updating optuna study ',study_name)
     


if __name__=='__main__':
    main()

