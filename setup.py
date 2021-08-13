from setuptools import setup
from setuptools.extension import Extension


setup(name='deepRD',
      version='0.1',
      description='Reaction dynamics and deep learning tools',
      author='Mauricio J. del Razo',
      author_email='maojrs@gmail.com',
      url='',
      packages=['deepRD', 'deepRD.models'],
      test_suite='nose.collector',
      tests_require=['nose']
      )
