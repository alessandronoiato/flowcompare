"""CLI entry points: ``train``, ``sample``, ``eval``.

Each script is usable either as ``python -m flowcompare.scripts.train ...``
or (once installed) via console scripts declared in ``pyproject.toml``.
The goal is to keep the scripts thin: they parse arguments, pick a process
and backbone by name, and call the same library functions that tests and
notebooks use. No project-specific glue lives here.
"""
