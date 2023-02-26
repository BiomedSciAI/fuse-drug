#TODO: move to fuse
import os
from fuse.utils.file_io import read_simple_int_file, save_text_file
from os.path import abspath

from pathlib import PurePath
import shutil
import re
import click

DEPLOYMENT_PREFIX = 'DEPLOYED_'

def create(orig_code_path:str, sessions_base_dir:str, import_name:str, session_group_name='main', require_git_repo_base:bool=False):    
    """
    Copies all files inside a project directory (including code, configuration, etc.) into a new directory,
    (the ID is running integer )

    Fixes imports to make sure that the code is ran from the copied directory and not from the orig source dir.

    Args:
        orig_code_path: the base from which to copy the project
        sessions_base_dir: the directory in which sessions will be copied INTO
        import_name: the way that the section of the project is imported. For example: 'zed_dti.fully_supervised.dti_classifier'
        session_group_name: the session will be located at {sessions_base_dir}/{sessions_project_name}/{session num}/DEPLOYED_{code module name}
        require_git_repo_base: do we require that orig_code_path will be at the base of a git repo
    
    """
    assert isinstance(sessions_base_dir, str)

    orig_code_path = abspath(orig_code_path)    
    sessions_base_dir = abspath(sessions_base_dir)

    if not os.path.isdir(orig_code_path):
        raise Exception(f"can't find orig_code_path={orig_code_path}")
    if require_git_repo_base:
        _found_git_dir = os.path.isdir(os.path.join(orig_code_path, '.git'))
        _found_git_file = os.path.isfile(os.path.join(orig_code_path, '.git')) #happens in git submodules
        if not (_found_git_dir or _found_git_file):
            raise Exception(f'Expected a git repository but got: orig_code_path={orig_code_path}')
    
    #module_name = os.path.split(orig_code_path)[1]

    os.makedirs(sessions_base_dir, exist_ok=True)
    session_num = acquire_available_session_number(sessions_base_dir)

    new_session_dir = os.path.join(sessions_base_dir, session_group_name, f'{session_num}', DEPLOYMENT_PREFIX+import_name.split('.')[-1])

    copy_dir_recursively_and_fix_imports(
        #os.path.join(orig_code_path, module_name), 
        orig_code_path,
        new_session_dir,
        import_name,
        exclude_dirs = ['.git', '__pycache__'], 
    )
    
    save_text_file(os.path.join(new_session_dir, 'session_created'), str(session_num))
    print(f'Created session dir {new_session_dir}')
    return new_session_dir, session_num
        

def acquire_available_session_number(path):
    #TODO: make this multi thread/process safe

    available_session_num_fn = os.path.join(path, 'next_available_session_num')
    if not os.path.isfile(available_session_num_fn):
        save_text_file(available_session_num_fn, '100')

    available_session_num = read_simple_int_file(available_session_num_fn)
    save_text_file(available_session_num_fn, str(available_session_num+1))
    return available_session_num

def copy_dir_recursively_and_fix_imports(root_src_dir, root_target_dir, import_name:str, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []

    splt = import_name.split('.')
    to_import_name = DEPLOYMENT_PREFIX+splt[-1]
    #if len(splt)>1:
    #    to_import_name = '.'.join(splt[:-1])+'.'+to_import_name

    for src_dir, dirs, files in os.walk(root_src_dir):
        found_exclude = False
        for exc in exclude_dirs:
            if exc in PurePath(src_dir).parts:
                found_exclude = True
                break
        if found_exclude:
            continue

        dst_dir = src_dir.replace(root_src_dir, root_target_dir)
                
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                assert False 
            if file_.endswith('.py') and not file_.startswith('.'):
                copy_and_fix_imports_python_file(src_file, os.path.join(dst_dir,file_),
                                                 import_name, to_import_name)
            else:
                shutil.copy(src_file, dst_dir)

def copy_and_fix_imports_python_file(src_file, out_file, from_module_name, to_module_name):
    '''
    reads src_file, changes each line according to regext_list and saves into out_file
    :param src_file:
    :param out_file:
    :return:
    '''

    from_module_name = from_module_name.replace('.','\\.')

    out_lines = []
    with open(src_file,'r') as read_f:
        for line in read_f:           

            # detect and handle any "import [orig_path] ..."
            found = re.search('import\s+'+from_module_name, line)
            if found is not None:
                s = found.start()
                e = found.end()
                line = line[:s] + 'import ' + to_module_name + line[e:]

            # detect and handle any "from [orig_path] ..."
            found = re.search('from\s+' + from_module_name, line)
            if found is not None:
                s = found.start()
                e = found.end()
                line = line[:s] + 'from ' + to_module_name + line[e:]
            
            out_lines.append(line)
    with open(out_file, 'w') as write_f:
        write_f.writelines(out_lines)


@click.command()
@click.argument('project_dir')#, help='project directory, does not have to be at the root of the code repo. for example \home\someone\repos\image_classification\vanilla_transformer')
@click.argument('sessions_base_dir')#, help='the base directory is where new sessions will be created')
@click.option('--requiregit', '-r', is_flag=True, help="require that project_dir will be inside a git repo")
def main(project_dir:str, sessions_base_dir:str, requiregit:bool=False,):
    create(project_dir, sessions_base_dir, require_git_repo=requiregit)

if __name__=='__main__':
    main()

