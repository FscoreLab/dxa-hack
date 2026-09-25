"""Опись реестра признаков: python -m dxa_feat

Не `-m dxa_feat.registry`: модуль задвоится, и опись выйдет пустой.
"""

from dxa_feat.registry import describe

print(describe())
