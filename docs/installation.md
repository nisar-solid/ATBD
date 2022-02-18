## Install ATBD

Tested on macOS and Linux, not sure about Windows.

### 1. Install conda

```bash
mkdir -p ~/tools; cd ~/tools

# download, install and setup (mini/ana)conda
# for Linux, use Miniconda3-latest-Linux-x86_64.sh
# for macOS, opt 2: curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh -b -p ~/tools/miniconda3
~/tools/miniconda3/bin/conda init bash
```

Close and restart the shell for changes to take effect.

```
conda config --add channels conda-forge
conda install wget git tree mamba --yes
```

### 2. Install ATBD, ISCE-2, ARIA-tools and MintPy to `atbd` environment

#### Download source code

```bash
cd ~/tools
git clone https://gitlab.com/nisar-science-algorithms/solid-earth/ATBD.git
git clone https://github.com/isce-framework/isce2.git ~/tools/isce2/src/isce2
git clone https://github.com/aria-tools/ARIA-tools.git
git clone https://github.com/insarlab/MintPy.git
```

#### Create `atbd` environment and install pre-requisites

```bash
# create new environment
conda create --name atbd
conda activate atbd

# install dependencies with conda
mamba install --yes --file ATBD/requirements.txt --file MintPy/docs/requirements.txt --file ARIA-tools/requirements.txt

# install dependencies not available from conda
ln -s ${CONDA_PREFIX}/bin/cython ${CONDA_PREFIX}/bin/cython3
$CONDA_PREFIX/bin/pip install ipynb        # import functions from ipynb files
```

#### Setup

Create an alias `load_atbd` in `~/.bash_profile` file for easy activation, _e.g._:

```bash
alias load_atbd='conda activate atbd; source ~/tools/ATBD/docs/config.rc'
```

#### Test the installation

Run the following to test the installation:

```bash
load_atbd
ariaDownload.py -h
smallbaselineApp.py -h
```
