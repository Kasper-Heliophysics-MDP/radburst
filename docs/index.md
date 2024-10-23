# Welcome to RadBurst Documentation

This documentation provides a guide to the software developed for the MDP Heliophysics Research Team. The goal is to detect and classify solar radio bursts from spectrogram data collected by our LWA antennas.

Below are links to key sections of the documentation:

## Developer Guide

- **[Installation](guides/installation.md)**: Instructions for setting up the project on your local machine.
- **[Development Workflow](guides/dev_workflow.md)**: Best practices and guidelines for contributing to the project.
- **[Updating Documentation](guides/update_docs.md)**: Instructions for maintaining and updating documentation.

## Code Reference

- **[Reference Overview](code_reference/index.md)**: Overview of the codebase structure and main components.
- **[Utilities](code_reference/utils/utils.md)**: General utility functions for data handling and visualization.
- **[Preprocessing](code_reference/utils/preprocessing.md)**: Functions for data preprocessing and standardization.
- **[Detection](code_reference/detection/detection.md)**: Algorithms for detecting solar radio bursts.
- **[Classification](code_reference/classification/classification.md)**: CNN-based classification of solar radio bursts.

## Examples

- **[Examples](examples/examples.md)**:

## Repository Structure

    data/                   # Explanations and links to datasets and sample data
    docs/                   # Documentation (.md files)
    notebooks/              # Jupyter notebooks
        exploration/        # Notebooks for data exploration, testing ideas and analysis        
        evaluation/         # Notebooks evaluating models and algorithms
        examples/           # Notebooks demonstrating how to use code
    radburst/               # Main directory for code developement
        detection/          # Burst detection code
        classification/     # Burst classification code
        utils/              # Common utility functions (e.g. load, preprocess, etc.)
    requirements.txt        # Project dependencies (read by setup.py when "pip install -e ." is run)
    mkdocs.yml              # Configuration for MkDocs documentation
