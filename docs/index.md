# Welcome to RadBurst Documentation

This documentation provides a guide to the software developed for the MDP Heliophysics Research Team. The goal is to detect and classify solar radio bursts from spectrogram data collected by our LWA antennas.

Below are links to key sections of the documentation:

## Developer Guide

- **[Installation](guides/installation.md)**: Instructions for setting up the project on your local machine.
- **[Development Workflow](guides/dev_workflow.md)**: Best practices and guidelines for contributing to the project.
- **[Updating Documentation](guides/update_docs.md)**: Instructions for maintaining and updating documentation.

## Code Reference

- **[Reference Overview](code_reference/index.md)**: 
- **[Utilities](code_reference/utils/utils.md)**: 
- **[Detection](code_reference/detection/detection.md)**: 
- **[Classification](code_reference/classification/classification.md)**: 

## Examples

- **[Examples](examples/examples.md)**: 


## Repository Structure

    docs/                   # Documentation (.md files)
    site/                   # Documentation website
    draft/                  
        detection/          # Burst detection work
        classification/     # Burst classification work
        utils/              # Common utility functions (e.g. load, preprocess, etc.)
    requirements.txt        # Project dependencies (read by setup.py when "pip install -e ." is run)
    mkdocs.yml              # Documentation site settings
    

