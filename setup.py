from setuptools import setup

setup(name='deepRD',
      version='0.1',
      description='Reaction dynamics and deep learning tools',
      author='Mauricio J. del Razo',
      author_email='maojrs@gmail.com',
      url='',
      packages=['deepRD', 'deepRD.reactionModels', 'deepRD.integrators'],
      test_suite='nose.collector',
      tests_require=['nose']
      )
