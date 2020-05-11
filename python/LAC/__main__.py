# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件允许模块包以python -m lac方式直接执行。

Authors: huangdingbang(huangdingbang@baidu.com)
Date:    2019/09/28 21:00:01
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import sys
from lac.cmdline import main
sys.exit(main())
