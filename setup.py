from setuptools import setup

setup(
    name="predict_molecule_property",
    version="1.0.0",
    author='Houssam ALRACHID',
    author_email='houssam.alrachid@devoteam.com',
    description='The objective of this application uses a deep learning model to predict basic properties of a molecule using its fingerprint features and smiles strings.',
    packages=["servier"],
    entry_points='''
        [console_scripts]
        predict_molecule_property = servier.cli:main
        '''
)
