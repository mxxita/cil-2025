from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='monocular_depth',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A modular deep learning solution for monocular depth estimation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/monocular_depth',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'train_depth=monocular_depth.main:main',
        ],
    },
) 