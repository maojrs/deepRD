from setuptools import setup

setup(name='deepRD',
      version='0.1',
      description='Deep learning tools for reaction-diffusion dynamics',
      author='Mauricio J. del Razo',
      author_email='maojrs@gmail.com',
      url='',
      packages=[
      'deepRD', 
      'deepRD.reactionModels', 
      'deepRD.reactionIntegrators',
      'deepRD.diffusionIntegrators'],
      test_suite='nose.collector',
      tests_require=['nose']
      )
