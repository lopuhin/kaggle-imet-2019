from setuptools import setup

setup(
    name='imet',
    packages=['imet'],
    entry_points={
        'console_scripts': [
            'imet = imet.main:main',
            'make-submission = imet.make_submission:main',
            'make-folds = imet.make_folds:main',
        ],
    },
)
