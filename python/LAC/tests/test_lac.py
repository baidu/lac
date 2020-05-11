# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件测试LAC的分词及词性标注功能

Authors: huangdingbang(huangdingbang@baidu.com)
Date:    2019/09/29 21:00:01
"""

from LAC import LAC
from LAC._compat import strdecode
import os
import sys

os.environ['PYTHONIOENCODING'] = 'UTF-8'

lac = LAC()

for line in sys.stdin:
    line = strdecode(line.strip())
    result = lac.run(line)
    print(' '.join(result))

    sent, tag = lac.run(line, return_tag=True)
    result_list = ['(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tag)]
    print(''.join(result_list))
