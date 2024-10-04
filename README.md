# RadBurst - Solar Radio Burst Detection & Classification

This repository contains software developed for the MDP Heliophysics Research Team. We are focused on detecting and classifying solar radio bursts using spectrogram data collected by our LWA antennas.

For a comprehensive guide to this codebase, please refer to the [documentation](https://kasper-heliophysics-mdp.github.io/radburst/).

- [Installation](#installation)
- [Repository Structure](#repository-structure)


## Installation

1. **Clone the GitHub repository:**
```bash
git clone https://github.com/Kasper-Heliophysics-MDP/radburst.git
```
2. **Move into the project directory:**
```bash
cd draft
```
3. **Set up a virtual enviroment (to isolate dependencies):**
```bash
python3 -m venv .venv .
```
4. **Activate the virtual enviroment:**
```bash
. .venv/bin/activate
```
5. **Install the project and dependencies:**
```bash
pip install -e . 
```
This installs the project as a package (in "editable" mode), enabling you to import it throughout the codebase while reflecting changes made without needing to reinstall.


## Repository Structure

    docs/                   # Documentation (.md files)
    site/                   # Documentation website
    radburst/                  
        detection/          # Burst detection work
        classification/     # Burst classification work
        utils/              # Common utility functions (e.g. load, preprocess, etc.)
    requirements.txt        # Project dependencies (read by setup.py when "pip install -e ." is run)
    mkdocs.yml              # Documentation site settings