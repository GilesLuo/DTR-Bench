from setuptools import setup, find_packages

setup(
    name='DTRBench',
    version='0.0.1',
    author='Zhiyao Luo, Mingcheng Zhu',
    author_email='zhiyao.luo@eng.ox.ac.uk',
    description='DTR-Bench: An in silico Environment and Benchmark Platform for Reinforcement Learning Based '
                'Dynamic Treatment Regime',
    packages=find_packages(),
    install_requires=[
        "DTRGym",
        "optuna=3.2.0",
    ],
)
