# Updating Documentation

This guide provides instructions for how to update and maintain project documentation. These steps explain how to make changes, preview them and deploy updates to the documentation site.

The documentation site is built using [MkDocs](https://www.mkdocs.org/). MkDocs reads the [Markdown](https://www.markdownguide.org/) (.md) files from the `docs/` folder and configures the site according to the settings defined in `mkdocs.yml`.

## 1. Edit Documentation Files

- Modify `.md` files in the `docs/` directory.
- For Markdown (`.md`) syntax, refer to the [Markdown Guide](https://www.markdownguide.org/)

## 2. Code Reference

- To automatically generate documentation from your code's doctrings (in modules, functions or classes), use the following syntax in your Markdown files:

```markdown
::: radburst.utils.preprocessing
```

- The line above will read the docstrings in `preprocessing.py` and create structured documentation that explains the functions, their parameters, and return values.
- The current `mkdocs.yml` is configured to use Google-style dosctrings. Here's an example of a Google-style docstring for a function:

    ```python
    def example_function(param1, param2):
        """Short description of the function.

        Longer description that provides more detail about what the function does,
        how it operates, and any important considerations. This can include
        information about the parameters, return values, and any exceptions that
        might be raised.

        Args:
            param1 (int): The first parameter to be processed.
            param2 (float): The second parameter to be processed.

        Returns:
            bool: True if the operation is successful, False otherwise.
        """
        return True
    ```

## 3. Updating `mkdocs.yml`

- The `mkdocs.yml` file in the root directory configures the documentation site. Changes to documentation files/structure will need to be reflected here to make sure the site is rendered correctly.
- Here is a preview of what it looks like:

    ```yaml
    site_name: RadBurst Documentation
    site_url: https://Kasper-Heliophysics-MDP.github.io/radburst

    repo_name: Kasper-Heliophysics-MDP/radburst
    repo_url: https://github.com/Kasper-Heliophysics-MDP/radburst

    nav:
    - Home: index.md
    - Developer Guide:
        - Installation: guides/installation.md
        - Development Workflow: guides/dev_workflow.md
        - Updating Documentation: guides/update_docs.md
    - Code Reference:
        - code_reference/index.md
    ```

## 4. Preview Changes Locally

- To preview your changes, run the following command from the project directory:

```bash
mkdocs serve
```
- This will provide a link to view the current documentation in your web browser.

## 5. Build the Documentation Site (Optional)

- To generate the static site files without deploying, run:

```bash
mkdocs build
```
- This will create site files in `site/`. Note `site/` is in `.gitignore` and this build command is automatically run in the following deployment step. Therefore, if changes look good after previewing with `mkdocs serve`, building is not required.

## 6. Deploy the Changes

- Once you're satisfied with your changes, they can be deployed:

```bash
mkdocs gh-deploy
```
- This will build the documentation (if not already built), commit changes to the `gh-pages` branch and push the `gh-pages` branch to GitHub. The GitHub repo will host the documentation using `gh-pages`.