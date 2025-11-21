import io
import os
import re # 引入 re 模块
from setuptools import setup, find_packages

def get_version():
    here = os.path.abspath(os.path.dirname(__file__))
    # 使用正则表达式从 _version.py 文件中安全读取版本号
    with open(os.path.join(here, 'neurodeckit', '_version.py'), encoding='utf-8') as f:
        version_file = f.read()
    
    # 查找 __version__ = "X.Y.Z"
    version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]*)['\"]", version_file, re.M)
    
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# 读取版本号
version_ns = {'__version__': get_version()}

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
