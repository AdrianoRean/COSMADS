import numpy as np
import time

class GetCurrentDataFromChip:
    seed = 0
    delay = 0.05
    cumulative_rotations = 0
    refresh_rate = 0
    last_execution_time = 0
    last_value = None
    call_parameters_list = ['chip_id']
    description = {
        "name": "GetCurrentDataFromChip",
        "brief_description": "Data service that, given a chip id of a chip embedded into a diecutter, provides the current speed, temperature and number of rotations measured by that chip.",
        "detailed_description": 
        """Data service that, given a chip id of a chip embedded into a diecutter, provides the current speed, temperature and number of rotations measured by that chip.
        In general each diecutter has exactly one chip embedded into it, and each chip is uniquely identified by an integer id. 
        This means that the chip id has a one-to-one correspondence with the id of the diecutter, but those two ids may not be the same.
        This data service takes a single parameter, namely the id of the chip (an integer) and returns a single dictionary containing the current speed, temperature and number of rotations of the diecutter it is attached to.
        The keys of the dictionary are "speed", "temperature" and "rotations".""",
        "usage_example":
        """If the id of the chip is 123, then the data service would be called as follows:
        chip_id = 123
        current_data = GetCurrentData.call(chip_id=123)
        # get the current speed, temperature and number of rotations
        current_speed = data["speed"]
        current_temperature = data["temperature"]
        current_rotations = data["rotations"]
    
        Things to keep in mind:
        - The refresh rate of the data is 1 second, i.e. the data is updated every second, so if the data service is called multiple times within a second, it will return the same value.
        - Note that the values in the dictionary represents the current values of the speed, etc. of the diecutter, they are not aggregated values.
        - Remember that the output value is a dictionary with the above specified keys, so avoid trying to access it using keys not in the above example.""",
        "input_parameters": ["chip_id:int"],
        "output_values": ["diecutter_current_data:dict"],
        "module": "diecutter"
    }

    def call(chip_id: int) -> dict:
        # add a small delay to simulate the time it takes to get the data from the chip
        #time.sleep(GetCurrentDataFromChip.delay)
        current_execution_time = time.time()
        # check if the refresh rate has passed or if it is the first time the function is called
        if GetCurrentDataFromChip.last_value is None or current_execution_time - GetCurrentDataFromChip.last_execution_time >= GetCurrentDataFromChip.refresh_rate:
            # set the random seed
            np.random.seed(GetCurrentDataFromChip.seed+chip_id)
            last_value = {
                "speed": np.random.randint(1, 10),
                "temperature": np.random.randint(0, 30),
                "rotations": np.random.randint(0, 100),
            }
            # update the seed
            GetCurrentDataFromChip.seed += 1
            # update the number of rotations
            GetCurrentDataFromChip.cumulative_rotations += last_value["rotations"]
            # update the last value
            GetCurrentDataFromChip.last_value = last_value
            # update the last execution time
            GetCurrentDataFromChip.last_execution_time = current_execution_time
            # update the last value field related to the rotation
            last_value["rotations"] = GetCurrentDataFromChip.cumulative_rotations
            return last_value
        else:
            return GetCurrentDataFromChip.last_value