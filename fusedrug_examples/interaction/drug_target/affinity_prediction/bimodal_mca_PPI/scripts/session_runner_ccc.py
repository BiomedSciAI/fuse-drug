from typing import Optional
from os.path import join, dirname, abspath
from fusedrug.utils.session_manager import session_creator
import click
import getpass
import subprocess
from fuse.utils.file_io import save_text_file
from cvar_pyutils.ccc import submit_job
from cvar_pyutils.ccc import submit_dependant_jobs
import colorama
colorama.init(autoreset=True)
from colorama import Fore, Back, Style
from zed_dti.fully_supervised.dti_classifier.utils import get_local_timestamp
from hydra import compose, initialize
from omegaconf import OmegaConf
from zed_dti.fully_supervised.dti_classifier.utils import hydra_resolvers, get_valid_filename

def run(
    script_relative_path:str='./run.py',
    sessions_base_dir:str=None, 
    only_create_session_dir:bool=False, 
    rolling_jobs:int=1, 
    use_existing_session_dir:str=None, 
    continue_from_checkpoint_from_the_first_job:bool=False,
    depends_on_job_id:str=None,
    session_group_name:str=None,
    run_cli_args:str='',
    pyutils_run_kwargs_override:Optional[dict]=None,
    ):
                  
    if use_existing_session_dir is None:
        if continue_from_checkpoint_from_the_first_job:
            raise Exception('--continue-from-checkpoint-from-the-first-job can only be used when combined with --use-existing-session-dir')

        if session_group_name is None:
            #extracting session_group_name from the config
            for name,func in hydra_resolvers.items():
                OmegaConf.register_new_resolver(name, func)

            initialize(config_path="../configs") #, job_name="test_app")    
            cfg = compose(config_name="train_config") # remember that we can provide overrides: overrides=["db=mysql", "db.user=me"]
            OmegaConf.resolve(cfg) 
            
            session_group_name = cfg['session_group_name']
            print(f'Since no "session_group_name" was provided, deduced it from "train_config.yaml": {session_group_name}')
            session_group_name = get_valid_filename(session_group_name)
            print(f'corrected (if needed) to linux file system friendly string: {session_group_name}')  

        if sessions_base_dir is None:
            sessions_base_dir = f'/dccstor/fmm/users/{getpass.getuser()}/dev/sessions/dti'
        project_dir = dirname(abspath(join(__file__,'../')))
        use_session_dir, session_num = session_creator.create(project_dir, sessions_base_dir, session_group_name=session_group_name, import_name='zed_dti.fully_supervised.dti_classifier')
    else:
        
        if sessions_base_dir is not None:
            raise Exception('You cannot use both "--already-existing-session-dir" and "--session-base-dir"')

        if only_create_session_dir:
            raise Exception('You cannot use both "--already-existing-session-dir" and "--only-create-session-dir"')

        print(f'using provided already existing session dir (not creating a new one) - {use_existing_session_dir}')
        use_session_dir = use_existing_session_dir


    if only_create_session_dir:
        print(f'only_create_session_dir flag requested! Exiting. You can run it directly by executing:\n'
        f'python {join(use_session_dir,script_relative_path)} {run_cli_args}\n'
        'or debug it in vscode by running:\n',
        f'python -m debugpy --listen 0.0.0.0:1326 --wait-for-client {join(use_session_dir,script_relative_path)} {run_cli_args}\n'
        )
        return

    assert isinstance(rolling_jobs, int) and (rolling_jobs >= 1)
    
    depend = None
    if depends_on_job_id:
        depend = [f'ended({depends_on_job_id})']

    shared_args = dict(        
        conda_env='bio',
        machine_type='x86', 
        #duration='24h', ### RESTORE
        duration='6h', ### DEBUG!
        #duration='1h', ### DEBUG!
        num_node=1, 
        num_cores=8, 
        num_gpus=1, 
        mem='300g', 
        #gpu_type='v100 | k80',  ### a100_80gb
        # gpu_type='a100_80gb',  
        depend=depend,
        #mail_log_file_when_done='username@ibm.com',
        verbose_output = True,
    )
    
    if pyutils_run_kwargs_override is not None:
        shared_args.update(pyutils_run_kwargs_override)


    if shared_args['duration'] != '24h':
        print(Fore.RED + f'!!! Warning: requested duration is not 24h (it is {shared_args["duration"]}). This is ok if it is intentional.')
    
    add_to_first = ''
    if continue_from_checkpoint_from_the_first_job:
        print('since --continue-from-checkpoint-from-the-first-job was requested, all jobs (including the first!) will load from an existing checkpoint.')
        add_to_first = ' load_from_checkpoint=last '

    if 1==rolling_jobs:
        job_id, jbsub_output = submit_job(command_to_run=f'python -u {join(use_session_dir,script_relative_path)}'+add_to_first+' '+run_cli_args,
            **shared_args)
        all_job_ids = [job_id]
        all_job_outputs = [jbsub_output]
    else:
        all_job_ids, all_job_outputs = submit_dependant_jobs(number_of_rolling_jobs=rolling_jobs, 
            command_to_run=[
                f'python -u {join(use_session_dir,script_relative_path)}'+' '+add_to_first+' '+run_cli_args,
                f'python -u {join(use_session_dir,script_relative_path)} load_from_checkpoint=last'+' '+run_cli_args,
            ],
            **shared_args)

    with open(join(use_session_dir,f'ccc_jobs_info@{get_local_timestamp("Israel")}.txt'),'w') as f:
        f.write(f'{len(all_job_ids)} jobs\n')
        for job_id, job_output in zip(all_job_ids, all_job_outputs):
            f.write(f'job_id:{job_id}\n')
            f.write(f'job_output:{job_output}\n')
    
@click.command()
@click.option('--script-relative-path', default='./run.py', help='path of the script to run, relative to the session dir. For example: "./run.py" or "./scripts/optuna_worker.py"')
@click.option('--sessions-base-dir','-s', help='sessions base directory, new sessions will be created here. Defaults to /dccstor/fmm/users/{USENAME}/dev/sessions')
@click.option('--only-create-session-dir', '-o', is_flag=True, help='a dry run in which the session is created, the (sub)project files are copied and the lsf (for ccc) command is printed but not actually executed.')
@click.option('--rolling-jobs', '-r', default=1, type=int, help='Run continuous jobs, each dependant of the previous')
@click.option('--use-existing-session-dir','-e', default=None, help='run on an already existing session - session_creator will not be used in this case.')
@click.option('--continue-from-checkpoint-from-the-first-job', '-c', is_flag=True, help='can be used only when combined with --use-existing-session-dir, it means that there will be no training from scratch, it will continue the existing session dir last store checkpoint. ')
@click.option('--depends-on-job-id','-d', default=None, help='job_id that needs to end before the requested job(s) begin')
@click.option('--session-group-name',default=None, help='group of the session - all session with the same session_group_name will be under the same sub dir, which makes automation/scripts easier later.\n If None is provides (the default) it will be deduced from "session_group_name" from train_config.yaml')
@click.option('--run-cli-args','-a', default='', help='args to send to run.py (which is hydra based) - make sure to wrap it with quotes. for example --run-cli-args "model.lr=0.00004 model.dropout=0.1"')
def main_cli(
    script_relative_path:str='./run.py',
    sessions_base_dir:str=None, 
    only_create_session_dir:bool=False, 
    rolling_jobs:int=1, 
    use_existing_session_dir:str=None, 
    continue_from_checkpoint_from_the_first_job:bool=False,
    depends_on_job_id:str=None,
    session_group_name:str=None,
    run_cli_args:str='',
    ):
    run(
        script_relative_path=script_relative_path,
        sessions_base_dir=sessions_base_dir,
        only_create_session_dir=only_create_session_dir,
        rolling_jobs=rolling_jobs,
        use_existing_session_dir=use_existing_session_dir,
        continue_from_checkpoint_from_the_first_job=continue_from_checkpoint_from_the_first_job,
        depends_on_job_id=depends_on_job_id,
        session_group_name=session_group_name,
        run_cli_args=run_cli_args,
    )
 
if __name__=='__main__':
    main_cli()





