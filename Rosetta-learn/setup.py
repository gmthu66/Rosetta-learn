from setuptools import setup

setup(name='Rosetta_learn',
      version='0.1',
      description='Recommender system for protein sequence design and optimization using Rosetta',
      url='https://github.com/tamimeur/Rosetta-learn.git',
      author='Tileli Amimeur',
      author_email='tamimeur@gmail.com',
      license='MIT',
      packages=['Rosetta_learn'],
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=[
          'pandas','numpy','click','keras==1.2','sklearn','scipy','xlrd','h5py',
      ],
      zip_safe=False,
      entry_points={
        'console_scripts': [
            'Rosetta-learn = Rosetta_learn.cli:main',
        ],
      })
