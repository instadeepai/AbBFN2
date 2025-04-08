import sys

import hydra
from omegaconf import DictConfig


class Singleton(type):
    """Singleton metaclass.

    As described here:
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """Create initial instance of class if it doesn't exist in _instances."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=Singleton):
    """Config singleton class."""

    _dict_config_object: DictConfig | None = None

    def __new__(cls, **kwargs) -> DictConfig:
        """Called every time a new instance is created.

        Only once because it is a singleton.
        """
        cls.initialize(**kwargs)
        return cls._dict_config_object

    @classmethod
    def initialize(
        cls, config_name="unconditional", config_path="../../experiments/configs"
    ) -> None:
        """Initializes or re-initializes the config."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(version_base=None, config_path=config_path)

        # We still need this overwriting because the config value is used in
        # get_gpus() which is called in global scope.
        # overrides = (
        #     ["raise_exception_if_gpus_not_available=false"]
        #     if "pytest" in sys.modules
        #     else []
        # )

        new_config = hydra.compose(config_name=config_name, overrides=[])

        if cls._dict_config_object is None:
            cls._dict_config_object = new_config
        else:
            cls().update(new_config)


# When running the tests, make sure that the original config is at least once
# loaded before the test config is then initialized. Otherwise, the __new__ method
# would never be called, which leads to recursion problems.
if "pytest" in sys.modules:
    Config()