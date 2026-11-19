"""Parallel direct ("wide skip") pathway for MTTGNet."""

__all__ = ["WideSkip", "MTTGNetWideWrapper", "build_clean_model"]


def __getattr__(name):
    if name in __all__:
        from experiments.mttgnet_v2 import wide

        return getattr(wide, name)
    raise AttributeError(name)
