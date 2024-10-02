from setuptools import setup, find_packages

def read_requirements(file_name):
    with open(file_name) as f:
        return [line.strip() for line in f if line and not line.startswith("#")]

setup(
    name='draft',
    version='0.1',
    packages=find_packages(),
    install_requires=read_requirements('requirements.txt'),
)