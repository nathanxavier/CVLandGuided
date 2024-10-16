from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

setup(
    name='featup',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'tqdm',
        'torchmetrics',
        'scikit-learn',
        'numpy',
        'matplotlib',
        'timm',
    ],
    author='Nathan Xavier',
    author_email='nathanxavier@ufmg.br',
    description='CVSegGuide',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nathanxavier/CVSegGuide',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    cmdclass={
        'build_ext': BuildExtension
    }
)
