from setuptools import setup, find_packages

setup(
    name='CVSegGuide',
    version='0.1.0',
    author='Nathan Xavier',
    author_email='nathanxavier@ufmg.br',
    description='Paper in review',
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
    url='https://github.com/nathanxavier/CVSegGuide',
    python_requires='>=3.8',
)
