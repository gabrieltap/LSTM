from setuptools import find_packages, setup

setup(
    name='LSTM',
    version='v1.0',
    packages=find_packages(exclude='tests'),
    install_requires=[
        'torch',
        'pandas',
        'numpy',
        'matplotlib'
    ],
    url='',
    license='',
    author='gabrip',
    author_email='gabriel.tap@gmail.com',
    description='LSTM neural network'
)
