from setuptools import setup, find_packages

def get_requirements():
    with open('requirements.txt') as req_file:
        requirements = req_file.readlines()
    requirements = [req.strip() for req in requirements]
    if '-e .' in requirements:
        requirements.remove('-e .')



setup(
    name="lesmills model",
    version='0.0.1',
    description='this is the randomforest model for memebr at lesmills',
    author='keyvan salehi',
    author_email='keyvan.salehi@lesmills.co.nz',
    packages=find_packages(),
    install_requires=get_requirements()
)    