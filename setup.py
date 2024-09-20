from setuptools import setup, find_packages

with open("README.md", "r", errors="ignore", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='NeuroDecKit',
    version='0.1',  
    author='LC.Pan',
    author_email='panlincong@tju.edu.cn.com',
    description='All Tools for MI-EEG Decoding',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/TJU-PanLC/NeuroDecKit',
    packages=find_packages(),  
    install_requires=[
        'braindecode',
        'catboost==1.2.5',
        'einops==0.8.0',
        'geoopt==0.5.0',
        'h5py',
        'joblib==1.2.0',
        'lightgbm==4.4.0',
        'mne==1.4.2',
        'moabb==1.1.0',
        'numpy==1.23.5',
        'pandas==1.5.3',
        'pooch==1.8.1',
        'psutil==5.9.5',
        'pynvml==11.5.0',
        'pyriemann==0.6',
        'scikit_learn==1.5.0',
        'scipy==1.14.0',
        'skorch==0.13.0',
        'statsmodels==0.14.0',
        'torch==2.0.1',
        'torchsummary==1.5.1',
        'xgboost==2.1.0',
    ],  
    keywords=['python', 'package'],
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: BSD 3-Clause License",
        "Operating System :: OS Independent",
    ],
)