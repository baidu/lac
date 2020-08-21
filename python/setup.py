# -*- coding: UTF-8 -*-
################################################################################
#
#   Copyright (c) 2020  Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

"""
Setup script.
"""

from setuptools import setup
import pkg_resources
from io import open

# 判断paddle安装版本，对版本进行设置
install_requires = []
try:
    import paddle

    # 若版本太低，设置版本的更新
    if paddle.__version__ < '1.6.0':
        installed_packages = pkg_resources.working_set
        paddle_pkgs = [i.key for i in installed_packages if "paddle" in i.key]

        if "paddlepaddle-gpu" in paddle_pkgs:
            install_requires = ['paddlepaddle-gpu>=1.6']
        elif "paddlepaddle-tiny" in paddle_pkgs:
            install_requires = ['paddlepaddle-tiny>=1.6']
        else:
            install_requires = ['paddlepaddle>=1.6']


except ImportError:
    install_requires = ['paddlepaddle>=1.6']


with open("README.md", "r", encoding='utf8') as fh:
   long_description = fh.read()

setup(
    name="LAC",
    version="2.0.5",
    author="Baidu NLP",
    author_email="nlp-fenci@baidu.com",
    description="A chinese lexical analysis tool by Baidu NLP.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/baidu/lac",
    install_requires=install_requires,
    packages=['LAC'],
    package_dir={'LAC': 'LAC'},
    package_data={'LAC': ['*.py', 'lac_model/*/*', 'seg_model/*/*']},
    platforms="any",
    license='Apache 2.0',
    keywords=('lac chinese lexical analysis'),
    classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
    ],
)
