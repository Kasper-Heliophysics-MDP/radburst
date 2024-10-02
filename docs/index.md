# Welcome to the docs!



## Installation

1. **Clone the GitHub repository:**
```bash
git clone https://github.com/Kasper-Heliophysics-MDP/draft.git
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



## Updating Documentation

This documentation site is built using [MkDocs](https://www.mkdocs.org/). MkDocs reads the [Markdown](https://www.markdownguide.org/) (.md) files from the `docs/` folder and configures the site according to the settings defined in `mkdocs.yml`.


1. **Edit Documentation Files:**
    Modify `.md` files in the `docs/` directory.

2. **Preview Changes Locally:**
    To preview your changes, run the following command from the project directory:
    ```bash
    mkdocs serve
    ```
    This command will provide a link to view the current documentation in your web browser.

3. **Build the Static Site:**
    ```bash
    mkdocs build
    ```

4. **Deploy the Changes:**
```bash
mkdocs gh-deploy
```
This command will build the docs, commit them to the `gh-pages` branch and push the `gh-branch` pages branch to GitHub.


## Repository Structure

    docs/                   # Documentation (.md files)
    site/                   # Documentation website
    draft/                  
        detection/          # Burst detection work
        classification/     # Burst classification work
        utils/              # Common utility functions (e.g. load, preprocess, etc.)
    requirements.txt        # Project dependencies (read by setup.py when "pip install -e ." is run)
    mkdocs.yml              # Documentation site settings
    

