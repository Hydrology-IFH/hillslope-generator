## Hillsope generator
Generates an artificial hillslope. The following hillslope shapes are
available:
- straight
- concave
- noisy

## Usage
**First step:**
Download the repository and install required Python packages in an environment:
```bash
git clone https://github.com/schwemro/hillslope_generator.git
cd hillslope_generator
conda env create -f conda-environment.yml
```
IMPORTANT: Add the folder containing the package to your PYTHONPATH! Modify
your .bashrc-file.

**Second step:**
Activate the anaconda environment and and move into the direcory of
hillslope_generator:
```bash
conda activate hillgen
python generate_hillslope.py --write-output True
```
