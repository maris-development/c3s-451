from enum import Enum

class Variable:
    class CMIP5Monthly(Enum):
        temperature_2m = 1 # 2 meter temperature
        
        def cds_name(self) -> str:
            """
            Get the corresponding CDS variable name.

            Returns:
                str: The CDS variable name.
            """
            translation_dict = {
                Variable.CMIP5Monthly.temperature_2m: '2m_temperature',
            }
            return translation_dict[self]
        
        def column_name(self) -> str:
            """
            Get the corresponding column name for dataframes.

            Returns:
                str: The column name.
            """
            translation_dict = {
                Variable.CMIP5Monthly.temperature_2m: 't2m',
            }
            return translation_dict[self]

    class CMIP6(Enum):
        near_surface_air_temperature = 1 # near surface air temperature
        daily_maximum_near_surface_air_temperature = 2 # maximum daily near surface air temperature
        daily_minimum_near_surface_air_temperature = 3 # minimum daily near surface air temperature
        precipitation = 2 # precipitation
        
        def cds_name(self) -> str:
            """
            Get the corresponding CDS variable name.

            Returns:
                str: The CDS variable name.
            """
            translation_dict = {
                Variable.CMIP6.near_surface_air_temperature: '2m_temperature',
                Variable.CMIP6.daily_maximum_near_surface_air_temperature: '2m_temperature',
                Variable.CMIP6.daily_minimum_near_surface_air_temperature: '2m_temperature',
                Variable.CMIP6.precipitation: 'total_precipitation',
            }
            return translation_dict[self]
        
        def column_name(self) -> str:
            """
            Get the corresponding column name for dataframes.

            Returns:
                str: The column name.
            """
            translation_dict = {
                Variable.CMIP6.near_surface_air_temperature: 'tas',
                Variable.CMIP6.daily_maximum_near_surface_air_temperature: 'tasmax',
                Variable.CMIP6.daily_minimum_near_surface_air_temperature: 'tasmin',
                Variable.CMIP6.precipitation: 'tp',
            }
            return translation_dict[self]
        
    class ERA5DailySingleLevel(Enum):
        temperature_2m_mean = 1 # 2 meter temperature mean
        temperature_2m_max = 2 # 2 meter temperature max
        temperature_2m_min = 3 # 2 meter temperature min
        total_precipitation = 4 # total precipitation
        mean_sea_level_pressure = 5 # mean sea level pressure
        
        def cds_name(self) -> str:
            """
            Get the corresponding CDS variable name.

            Returns:
                str: The CDS variable name.
            """
            translation_dict = {
                Variable.ERA5DailySingleLevel.temperature_2m_mean: '2m_temperature',
                Variable.ERA5DailySingleLevel.temperature_2m_max: '2m_temperature',
                Variable.ERA5DailySingleLevel.temperature_2m_min: '2m_temperature',
                Variable.ERA5DailySingleLevel.total_precipitation: 'total_precipitation',
                Variable.ERA5DailySingleLevel.mean_sea_level_pressure: 'mean_sea_level_pressure',
            }
            return translation_dict[self]
        
        def cds_daily_statistic(self) -> str:
            """
            Get the corresponding CDS daily statistic name.

            Returns:
                str: The CDS daily statistic name.
            """
            translation_dict = {
                Variable.ERA5DailySingleLevel.temperature_2m_mean: 'daily_mean',
                Variable.ERA5DailySingleLevel.temperature_2m_max: 'daily_maximum',
                Variable.ERA5DailySingleLevel.temperature_2m_min: 'daily_minimum',
                Variable.ERA5DailySingleLevel.total_precipitation: 'daily_sum',
                Variable.ERA5DailySingleLevel.mean_sea_level_pressure: 'daily_mean',
            }
            return translation_dict[self]
        
        def cds_variable_renames(self) -> dict[str, str]:
            """
            Get the renaming dictionary for CDS variable names to standard names.

            Returns:
                dict: A dictionary mapping CDS variable names to standard names.
            """
            translation_dict = {
                Variable.ERA5DailySingleLevel.temperature_2m_mean: {'t2m': 't2m'},
                Variable.ERA5DailySingleLevel.total_precipitation: {'tp': 'tp'},
                Variable.ERA5DailySingleLevel.temperature_2m_min: {'t2m': 't2m'},
                Variable.ERA5DailySingleLevel.temperature_2m_max: {'t2m': 't2m'},
            }
            return translation_dict[self]
        
        def column_name(self) -> str:
            """
            Get the corresponding column name for dataframes.

            Returns:
                str: The column name.
            """
            translation_dict = {
                Variable.ERA5DailySingleLevel.temperature_2m_mean: 't2m',
                Variable.ERA5DailySingleLevel.temperature_2m_max: 't2m', 
                Variable.ERA5DailySingleLevel.temperature_2m_min: 't2m', 
                Variable.ERA5DailySingleLevel.total_precipitation: 'tp',
                Variable.ERA5DailySingleLevel.mean_sea_level_pressure: 'msl',
            }
            return translation_dict[self]
        
        def beacon_name(self) -> str:
            """
            Get the corresponding Beacon variable name.

            Returns:
                str: The Beacon variable name.
            """
            translation_dict = {
                Variable.ERA5DailySingleLevel.temperature_2m_mean: 't2m',
                Variable.ERA5DailySingleLevel.temperature_2m_max: 't2m_max',
                Variable.ERA5DailySingleLevel.temperature_2m_min: 't2m_min',
                Variable.ERA5DailySingleLevel.total_precipitation: 'total_precipitation',
                Variable.ERA5DailySingleLevel.mean_sea_level_pressure: 'msl',
            }
            return translation_dict[self]
        
        def beacon_alias(self) -> str:
            """
            Get the corresponding Beacon alias for the variable.

            Returns:
                str: The Beacon alias.
            """
            translation_dict = {
                Variable.ERA5DailySingleLevel.temperature_2m_mean: 't2m',
                Variable.ERA5DailySingleLevel.temperature_2m_max: 't2m',
                Variable.ERA5DailySingleLevel.temperature_2m_min: 't2m',
                Variable.ERA5DailySingleLevel.total_precipitation: 'tp',
                Variable.ERA5DailySingleLevel.mean_sea_level_pressure: 'msl',
            }
            return translation_dict[self]
        
        def beacon_variable_renames(self) -> dict[str, str]:
            """
            Get the renaming dictionary for CDS variable names to standard names.

            Returns:
                dict: A dictionary mapping CDS variable names to standard names.
            """
            translation_dict = {
                Variable.ERA5DailySingleLevel.temperature_2m_mean: {'t2m': 't2m'},
                Variable.ERA5DailySingleLevel.total_precipitation: {'total_precipitation': 'tp'},
                Variable.ERA5DailySingleLevel.temperature_2m_min: {'t2m_min': 't2m'},
                Variable.ERA5DailySingleLevel.temperature_2m_max: {'t2m_max': 't2m'},
            }
            return translation_dict[self]
        
    class ERA5DailyPressureLevels(Enum):
        geopotential = 1 # geopotential
        
        def cds_name(self) -> str:
            """
            Get the corresponding CDS variable name.

            Returns:
                str: The CDS variable name.
            """
            translation_dict = {
                Variable.ERA5DailyPressureLevels.geopotential: 'geopotential',
            }
            return translation_dict[self]
        
        def beacon_name(self) -> str:
            """
            Get the corresponding Beacon variable name.

            Returns:
                str: The Beacon variable name.
            """
            translation_dict = {
                Variable.ERA5DailyPressureLevels.geopotential: 'z',
            }
            return translation_dict[self]
        
        def cds_daily_statistic(self) -> str:
            """
            Get the corresponding CDS daily statistic name.

            Returns:
                str: The CDS daily statistic name.
            """
            translation_dict = {
                Variable.ERA5DailyPressureLevels.geopotential: 'daily_mean',
            }
            return translation_dict[self]
    
        def column_name(self) -> str:
            """
            Get the corresponding column name for dataframes.

            Returns:
                str: The column name.
            """
            translation_dict = {
                Variable.ERA5DailyPressureLevels.geopotential: 'z',
            }
            return translation_dict[self]