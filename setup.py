#!/usr/bin/env python

# Python batteries
import os
from setuptools import find_packages, setup
from pip.download import PipSession
from pip.req import parse_requirements


project_dir = os.path.dirname(os.path.realpath(__file__))
packages_dir = os.path.join(project_dir, 'src')
requirements_path = os.path.join(project_dir, 'requirements.txt')
readme_path = os.path.join(project_dir, 'README.md')

install_reqs = parse_requirements(requirements_path, session=PipSession())
reqs = [str(ir.req) for ir in install_reqs]

def readme():
    with open(readme_path) as f:
        return f.read()
    #end
#end


setup(
    name='LearnAtariBoxing',
    version='0.2.0',
    url='https://gitlab.com/yunik1004/openai-gym-Boxing',
    author='INKYU PARK',
    author_email='yunik1004@gmail.com',
    description="This is the project for generating the learning agent of Atari 'Boxing' game using reinforcement learning and OpenAi-Gym.",
    long_description=readme(),
    python_requires='~=2.7',
    packages=find_packages(packages_dir),
    package_dir={'': packages_dir},
    install_requires=reqs,
)