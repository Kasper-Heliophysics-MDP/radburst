# Installation

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