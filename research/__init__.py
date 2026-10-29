"""research 包初始化。

作用：把项目根目录和 research 目录都加入 sys.path，
使 research/ 下的研究脚本既能 import 根目录的生产模块（yh_datasource 等），
也能 import 同目录的其他研究脚本。

必须用 -m 方式运行，例如：
    python -m research.yh_regime_calib6

直接 `python research/xxx.py` 不会执行本文件，会报 ModuleNotFoundError。
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)

for _p in (_parent, _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)
