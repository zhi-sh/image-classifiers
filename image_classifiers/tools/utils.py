# -*- coding: utf-8 -*-
# @DateTime :2021/2/23 下午4:57
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import importlib


def import_from_string(dotted_path):
    r'''根据点分的字符串路径，导入对应的模块'''
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = f"{dotted_path} doesn't look like a module path."
        raise ImportError(msg)

    module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = f"Module {module_path} does not define a {class_name} attribute/class."
        raise ImportError(msg)
