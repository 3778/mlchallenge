from setuptools import setup, find_packages

setup(name='challenge',
      version='0.1',
      description="3778's Machine Learning Challenge",
      url='https://github.com/3778/ml-challenge',
      author='3778 Data Science',
      author_email='datascience@3778.care',
      packages=find_packages(),
      install_requires = ['pandas>=0.25.2', 'pytest', 'sklearn', 'numpy'],
      zip_safe=False)
