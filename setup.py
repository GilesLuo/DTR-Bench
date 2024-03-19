from setuptools import setup, find_packages

setup(
    name='DTR-Bench',
    version='0.1.0',
    author='Zhiyao Luo, Mingcheng Zhu',
    author_email='zhiyao.luo@eng.ox.ac.uk',
    description='DTR-Bench: An in silico Environment and Benchmark Platform for Reinforcement Learning Based '
                'Dynamic Treatment Regime',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        "DTRGym",
        "optuna==3.2.0",
        "wandb",
    ],
)
