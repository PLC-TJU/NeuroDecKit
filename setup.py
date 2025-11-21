import io
import os
from setuptools import setup, find_packages

# 读取版本号
here = os.path.abspath(os.path.dirname(__file__))
version_ns = {}
if 'NEURODECKIT_VERSION' in os.environ:
    version_ns['__version__'] = os.environ['NEURODECKIT_VERSION']
else:
    with open(os.path.join(here, 'neurodeckit', '_version.py')) as vf:
        exec(vf.read(), version_ns)

print(f"Current version: {version_ns['__version__']}")

# 读取 README
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='neurodeckit',
    version=version_ns['__version__'],
    author='LC. Pan',
    author_email='panlincong@tju.edu.cn',
    description='Full chain toolkit for EEG signal decoding',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/PLC-TJU/NeuroDecKit',
    project_urls={
        'Documentation': 'https://github.com/PLC-TJU/NeuroDecKit#readme',
        'Source': 'https://github.com/PLC-TJU/NeuroDecKit',
        'Tracker': 'https://github.com/PLC-TJU/NeuroDecKit/issues',
    },
    packages=find_packages(exclude=('tests',)),
    python_requires='>=3.10',
    install_requires=[
        'braindecode',
        'einops',
        'geoopt',
        'h5py',
        'joblib',
        'mne',
        'numpy',
        'pandas',
        'pooch',
        'psutil',
        'pynvml',
        'pyriemann>=0.6.0',
        'scikit_learn',
        'scipy',
        'skorch',
        'statsmodels',
        'torch',
        'torchsummary',
    ],  
    keywords=['python', 'package'],
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    license='BSD-3-Clause',
)
