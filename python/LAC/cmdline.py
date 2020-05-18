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
本文件提供了命令行工具的入口逻辑。
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import argparse
parser = argparse.ArgumentParser(description='LAC Init Argments')
parser.add_argument('--segonly', action='store_true',
                    help='run segment only if setting')
args = parser.parse_args()

__all__ = [
    'main',
]


def main(args=args):
    """主程序入口"""
    from LAC import LAC
    from LAC._compat import strdecode
    import sys

    if args.segonly:
        lac = LAC(mode='seg')
    else:
        lac = LAC()

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        line = strdecode(line.strip())
        if args.segonly:
            print(u" ".join(lac.run(line)))
        else:
            words, tags = lac.run(line)
            print(u" ".join(u"%s/%s" % (word, tag)
                            for word, tag in zip(words, tags)))

    return 0
