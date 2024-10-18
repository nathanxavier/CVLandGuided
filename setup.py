from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

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
        'timm==1.0.9',
    ],
    url='https://github.com/nathanxavier/CVSegGuide',
    python_requires='>=3.8',
    ext_modules=[
        CUDAExtension(
            'adaptive_conv_cuda_impl',
            [
                'models/backbones/featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp',
                'models/backbones/featup/adaptive_conv_cuda/adaptive_conv_kernel.cu',
            ]),
        CppExtension(
            'adaptive_conv_cpp_impl',
            ['models/backbones/featup/adaptive_conv_cuda/adaptive_conv.cpp'],
            undef_macros=["NDEBUG"]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
