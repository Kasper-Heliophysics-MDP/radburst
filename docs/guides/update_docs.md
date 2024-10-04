# Updating Documentation

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