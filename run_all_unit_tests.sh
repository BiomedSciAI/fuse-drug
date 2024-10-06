#!/bin/bash
source ~/.bashrc

export CUDA_HOME=/opt/share/cuda-11.8 #to prevent mismatch between cuda runtime and the cuda version used to compile pytorch

# check if current env already exist
find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

# error message when failed to lock
lockfailed()
{
    echo "failed to get lock"
    exit 1
}

# create an environment including the set of requirements specified in the requirements (if not already exist)
create_env() {
    force_cuda_version=$1
    env_path=$2
    mode=$3
    start_time=`date +%s`

    requirements=$(cat ./requirements/requirements.txt)
    requirements+=$(cat ./requirements/requirements_dev.txt)

    if [ $mode = "examples" ]; then
        requirements+=$(cat ./fusedrug_examples/requirements.txt)
    fi

    # Python version
    PYTHON_VER=3.8
    ENV_NAME="fuse-drug_$PYTHON_VER-CUDA-$force_cuda_version-$(echo -n $requirements | sha256sum | awk '{print $1;}')"
    echo $ENV_NAME

    # env full name
    if [ $env_path = "no" ]; then
        env="-n $ENV_NAME"
    else
        env="-p $env_path/$ENV_NAME"
    fi


    # create a lock
    mkdir -p ~/env_locks # will create dir if not exist
    lock_filename=~/env_locks/.$ENV_NAME.lock
    echo "Lock filename $lock_filename"

    (
        flock -w 1200 -e 873 || lockfailed # wait for lock at most 20 minutes

        # Got lock - excute sensitive code
        echo "Got lock: $ENV_NAME"

        nvidia-smi

        if find_in_conda_env $ENV_NAME ; then
            echo "Environment exists: $env"
        else
            echo "Mode=$mode"
            # create an environment
            echo "Creating new environment: $env"
            conda create $env python=$PYTHON_VER -y
            echo "Creating new environment: $env - Done"
            
            echo "pipdeptree after basic env creation step"
            conda run -p $env --live-stream pipdeptree

            # install PyTorch
            if [ $force_cuda_version != "no" ]; then
                echo "forcing cudatoolkit $force_cuda_version"
                conda install $env pytorch torchvision pytorch-cuda=$force_cuda_version -c pytorch -c nvidia
                echo "forcing cudatoolkit $force_cuda_version - Done"
            fi

            echo "pipdeptree after install pytorch segment"
            conda run -p $env --live-stream pipdeptree

            echo "Installing FuseMedML"
            conda run $env --no-capture-output --live-stream pip install git+https://github.com/BiomedSciAI/fuse-med-ml@master
            echo "Installing FuseMedML - Done"

            echo "pipdeptree after install FuseMedML segment"
            conda run -p $env --live-stream pipdeptree


            echo "Installing core requirements"
            conda run $env --no-capture-output --live-stream pip install -r ./requirements/requirements.txt
            conda run $env --no-capture-output --live-stream pip install -r ./requirements/requirements_dev.txt
            echo "Installing core requirements - Done"

            echo "pipdeptree after install core requirements segment"
            conda run -p $env --live-stream pipdeptree

            if [ $mode = "examples" ]; then
                echo "Installing examples requirements"
                conda run $env --no-capture-output --live-stream pip install -r ./fusedrug_examples/requirements.txt
                echo "Installing examples requirements - Done"
            fi

            echo "pipdeptree after install examples segment"
            conda run $env --live-stream pipdeptree
        fi
    ) 873>$lock_filename

    # set env name
    ENV_TO_USE=$env

    end_time=`date +%s`
    echo "Created env $env in `expr $end_time - $start_time` seconds."
}


# create environment and run all unit tests
echo "input args ($#) - $@"

if [ "$#" -gt 0 ]; then
    force_cuda_version=$1
else
    force_cuda_version="no"
fi

if [ "$#" -gt 1 ]; then
    env_path=$2
else
    env_path="no"
fi

echo "Create core env"
create_env $force_cuda_version $env_path "core"
echo "Create core env - Done"

echo "Running core unittests in $ENV_TO_USE"
conda run $env --no-capture-output --live-stream python ./run_all_unit_tests.py core
echo "Running core unittests - Done"

echo "Create examples env"
create_env $force_cuda_version $env_path "examples"
echo "Create examples env - Done"

# update gitsubmodule - loading existing conda env doesn't solve it
# each different clone of the repo should be followed by submodules initialization
echo "Updating git submodules"
git submodule sync
git submodule update --init --recursive
echo "Updating git submodules - Done"

echo "Running examples unittests in $ENV_TO_USE"
conda run $env --no-capture-output --live-stream python ./run_all_unit_tests.py examples
echo "Running examples unittests - Done"
