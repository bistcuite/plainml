from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup (
    name='plainml',
    version='0.1',
    packages=['plainml', 'plainml.linear','plainml.preprocessing'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KaspiElectronics/plainml",
    install_requires=[
        'numpy',
        'pandas',
        'joblib',
        'matplotlib'
    ]
 )
