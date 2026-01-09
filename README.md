
# C3S Event Attirbution Tools

  This repository is for the Event Attribution Tools library for the C3S-451 project.

## Documentation:

  You can find the github pages documentation site [here](https://maris-development.github.io/c3s-451/latest/)

## Installation 

Install using PyPi:
`pip install c3s-event-attribution-tools`

Or install from GitHub directly:
`pip install git+https://github.com/maris-development/c3s-451`
  
 ## Usage
 
 For more examples please view the example notebooks in [the repository.](https://github.com/maris-development/c3s-451)

```python
from  datetime  import  datetime, timedelta
import  os
from  c3s_event_attribution_tools.data  import  DataClient

climate_data_store_token = 'aaaa-bbbb-cccc-dddd-eeee' #placeholder
your_save_directory = os.path.join(os.getcwd(), "../data")
bbox = (-105, 22, -100, 27)
event_end = datetime(2024, 6, 7)
event_start = event_end  -  timedelta(days=14)
parameter = "Tmean"
to_unit = "c"

gr_daily = DataClient(climate_data_store_token).fetch_data(parameter=parameter, bbox=bbox, time_range=(datetime(1951,1,1), event_end), to_unit=to_unit)

gr_daily.head()

```
  


