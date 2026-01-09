from enum import Enum


class Variable(Enum):
    tp = 1 # total precipitation
    t2mean = 2 # mean daily temperature at 2 meters
    t2min = 3 # min daily temperature at 2 meters
    t2max = 4 # max daily temperature at 2 meters
    mslp = 5 # mean sea level pressure
    geopotential = 6 # geopotential
    
    def cds_name(self) -> str:
        """
        Get the corresponding CDS variable name.

        Returns:
            str: The CDS variable name.
        """
        translation_dict = {
            Variable.t2mean: '2m_temperature',
            Variable.tp: 'total_precipitation',
            Variable.t2min: '2m_temperature',
            Variable.t2max: '2m_temperature',
            Variable.mslp: 'mean_sea_level_pressure',
            Variable.geopotential: 'geopotential',
        }
        return translation_dict[self]
    
    def cds_daily_statistic(self) -> str:
        """
        Get the corresponding CDS daily statistic name, if applicable.

        Returns:
            str | None: The CDS daily statistic name, or None if not applicable.
        """
        translation_dict = {
            Variable.t2mean: 'daily_mean',
            Variable.tp: 'daily_sum',
            Variable.t2min: 'daily_minimum',
            Variable.t2max: 'daily_maximum',
            Variable.mslp: 'daily_mean',
            Variable.geopotential: 'daily_mean',
        }
        return translation_dict[self]
    
    def cds_variable_renames(self) -> dict[str, str]:
        """
        Get the renaming dictionary for CDS variable names to standard names.

        Returns:
            dict: A dictionary mapping CDS variable names to standard names.
        """
        translation_dict = {
            Variable.t2mean: {'t2m': 't2m'},
            Variable.tp: {'tp': 'tp'},
            Variable.t2min: {'t2m': 't2m_min'},
            Variable.t2max: {'t2m': 't2m_max'},
        }
        return translation_dict[self]
    
    def beacon_name(self) -> str:
        """
        Get the corresponding Beacon variable name.

        Returns:
            str: The Beacon variable name.
        """
        translation_dict = {
            Variable.t2mean: 't2m',
            Variable.tp: 'tp',
            Variable.t2min: 't2m_min',
            Variable.t2max: 't2m_max',
        }
        return translation_dict[self]