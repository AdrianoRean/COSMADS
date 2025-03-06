import numpy as np
import random

class GetSerialCamera1IdFromDiecutterId:
    seed = 1
    call_parameters_list = ['diecutter_id']
    description = {
        "name": "GetSerialCamera1IdFromDiecutterId",
        "brief_description": "Data service that, given the id of a diecutter, provides the serial of the camera1 installed.",
        "detailed_description": 
        """Data service that, given the id of a diecutter, provides the serial of the camera1 installed.
        The data service relies on the management database of the cardboard factory to provide the camera1 serial.
        The data service takes a single parameter, namely the id of the diecutter (an integer) and returns a single integer, which is the serial of the camera1.""",
        "usage_example":
        """If the id of the diecutter is 123, then the data service would be called as follows:
        diecutter_id = 123
        serial_camera1 = GetSerialCamera1IdFromDiecutterId.call(diecutter_id=123)
        # assuming the serial is 4561245
        print(serial_camera1)  # 4561245

        Things to keep in mind:
        - The serial is an integer, so avoid trying to access it as a dictionary.""",
        "input_parameters": ["diecutter_id:int"],
        "output_values": ["serial_camera1:int"],
        "module": "database"
    }

    def call(diecutter_id: int) -> int:
        # set the random seed
        random.seed(GetSerialCamera1IdFromDiecutterId.seed+diecutter_id)
        # generate random serial camera1
        serial_camera1 = random.randrange(100000, 900000)
        return serial_camera1

class GetSerialCamera2IdFromDiecutterId:
    seed = 2
    call_parameters_list = ['diecutter_id']
    description = {
        "name": "GetSerialCamera2IdFromDiecutterId",
        "brief_description": "Data service that, given the id of a diecutter, provides the serial of the camera2 installed.",
        "detailed_description": 
        """Data service that, given the id of a diecutter, provides the serial of the camera2 installed.
        The data service relies on the management database of the cardboard factory to provide the camera2 serial.
        The data service takes a single parameter, namely the id of the diecutter (an integer) and returns a single integer, which is the serial of the camera2.""",
        "usage_example":
        """If the id of the diecutter is 123, then the data service would be called as follows:
        diecutter_id = 123
        serial_camera2 = GetSerialCamera2IdFromDiecutterId.call(diecutter_id=123)
        # assuming the serial is 4561245
        print(serial_camera2)  # 4561245

        Things to keep in mind:
        - The serial is an integer, so avoid trying to access it as a dictionary.""",
        "input_parameters": ["diecutter_id:int"],
        "output_values": ["serial_camera2:int"],
        "module": "database"
    }

    def call(diecutter_id: int) -> int:
        # set the random seed
        random.seed(GetSerialCamera2IdFromDiecutterId.seed+diecutter_id)
        # generate random serial camera2
        serial_camera2 = random.randrange(100000, 900000)
        return serial_camera2

class GetDiecuttersIdOfFactory:
    seed = 0
    call_parameters_list = []
    description = {
        "name": "GetDiecuttersIdOfFactory",
        "brief_description": "Data service that provides the list of id of the diecutters installed in the factory.",
        "detailed_description":
        """Data service that provides the id of the diecutter installed in the factory.
        The data service relies on the management database of the cardboard factory to provide the list of diecutter id.
        The data service takes no parameter and returns a list of integers, which are the id of the diecutters installed in the factory.""",
        "usage_example":
        """
        Example usage:
        - The data service would be called as follows:
        diecutters_id = GetDiecuttersIdOfFactory.call()
        # assuming the diecutters id are [1, 2, 3]
        print(diecutters_id)  # [1, 2, 3]

        Things to keep in mind:
        - The diecutters id is a list of integers, so avoid trying to access it as a dictionary.""",
        "input_parameters": [],
        "output_values": ["diecutters_id:list"],
        "module": "database"
    }

    def call() -> list:
        # set the random seed
        np.random.seed(GetDiecuttersIdOfFactory.seed)
        random.seed(GetDiecuttersIdOfFactory.seed)
        how_many_diecutters = np.random.randint(1, 10)
        # generate random diecutters id
        diecutters_id = [random.randrange(10000, 90000) for _ in range(how_many_diecutters)]
        return diecutters_id

class GetDiecutterChipIdFromDiecutterId:
    seed = 0
    call_parameters_list = ['diecutter_id']
    description = {
        "name": "GetDiecutterChipIdFromDiecutterId",
        "brief_description": "Data service that, given the id of a diecutter, provides the id of the chip embedded into that diecutter.",
        "detailed_description": 
        """Data service that, given the id of a diecutter, provides the id of the chip embedded into that diecutter.
        The data service relies on the management database of the cardboard factory to provide the chip id.
        The data service takes a single parameter, namely the id of the diecutter (an integer) and returns a single integer, which is the chip id of the diecutter.""",
        "usage_example":
        """
        - If the id of the diecutter is 123, then the data service would be called as follows:
        diecutter_id = 123
        chip_id = GetDiecutterChipIdFromDiecutterId.call(diecutter_id=123)
        # assuming the chip id is 456
        print(chip_id)  # 456

        Things to keep in mind:
        - The chip id is an integer, so avoid trying to access it as a dictionary.""",
        "input_parameters": ["diecutter_id:int"],
        "output_values": ["chip_id:int"],
        "module": "database"
    }

    def call(diecutter_id: int) -> int:
        # set the random seed
        np.random.seed(GetDiecutterChipIdFromDiecutterId.seed+diecutter_id)
        # generate random chip id
        chip_id = np.random.randint(0, 100)
        return chip_id
   

class GetCamera1IdFromDiecutterId:
    seed = 0
    call_parameters_list = ['diecutter_id']
    description = {
        "name": "GetCamera1IdFromDiecutterId",
        "brief_description": "Data service that, given the id of a diecutter, provides the id of the camera1 used to capture the frames of the cardboard cutouts produced by that diecutter.",
        "detailed_description": 
        """Data service that, given the id of a diecutter, provides the id of the camera1 used to capture the frames of the cardboard cutouts produced by that diecutter.
        The data service relies on the management database of the cardboard factory to provide the camera1 id.
        The data service takes a single parameter, namely the id of the diecutter (an integer) and returns a single integer, which is the id of the camera1 used to capture the frames of the cardboard cutouts produced by that diecutter.""",
        "usage_example":
        """If the id of the diecutter is 123, then the data service would be called as follows:
        diecutter_id = 123
        camera1_id = GetCamera1IdFromDiecutterId.call(diecutter_id=123)
        # assuming the camera1 id is 456
        print(camera1_id)  # 456

        Things to keep in mind:
        - The camera1 id is an integer, so avoid trying to access it as a dictionary.""",
        "input_parameters": ["diecutter_id:int"],
        "output_values": ["camera1_id:int"],
        "module": "database"
    }

    def call(diecutter_id: int) -> int:
        # set the random seed
        np.random.seed(GetCamera1IdFromDiecutterId.seed+diecutter_id)
        # generate random camera1 id
        camera1_id = np.random.randint(0, 100)
        return camera1_id
    

class GetCamera2IdFromDiecutterId:
    seed = 0
    call_parameters_list = ['diecutter_id']
    description = {
        "name": "GetCamera2IdFromDiecutterId",
        "brief_description": "Data service that, given the id of a diecutter, provides the id of the camera2 used to capture the frames of the cardboard cutouts produced by that diecutter.",
        "detailed_description": 
        """Data service that, given the id of a diecutter, provides the id of the camera2 used to capture the frames of the cardboard cutouts produced by that diecutter.
        The data service relies on the management database of the cardboard factory to provide the camera2 id.
        The data service takes a single parameter, namely the id of the diecutter (an integer) and returns a single integer, which is the id of the camera2 used to capture the frames of the cardboard cutouts produced by that diecutter.""",
        "usage_example":
        """If the id of the diecutter is 123, then the data service would be called as follows:
        diecutter_id = 123
        camera2_id = GetCamera2IdFromDiecutterId.call(diecutter_id=123)
        # assuming the camera2 id is 456
        print(camera2_id)  # 456

        Things to keep in mind:
        - The camera2 id is an integer, so avoid trying to access it as a dictionary.""",
        "input_parameters": ["diecutter_id:int"],
        "output_values": ["camera2_id:int"],
        "module": "database"
    }

    def call(diecutter_id: int) -> int:
        # set the random seed
        np.random.seed(GetCamera2IdFromDiecutterId.seed+diecutter_id)
        # generate random camera2 id
        camera2_id = np.random.randint(0, 100)
        return camera2_id


class GetCurrentSessionIdFromDiecutterId:
    seed = 0
    call_parameters_list = ['diecutter_id']
    description = {
        "name": "GetCurrentSessionIdFromDiecutterId",
        "brief_description": "Data service that, given the id of a diecutter, provides the current session id of that diecutter.",
        "detailed_description": 
        """Data service that, given the id of a diecutter, provides the current session id of that diecutter.
        The data service relies on the management database of the cardboard factory to provide the session id.
        The data service takes a single parameter, namely the id of the diecutter (an integer) and returns a single integer, which is the current session id of the diecutter.""",
        "usage_example":
        """If the id of the diecutter is 123, then the data service would be called as follows:
        diecutter_id = 123
        session_id = GetCurrentSessionIdFromDiecutterId.call(diecutter_id=123)
        # assuming the session id is 456
        print(session_id)  # 456

        Things to keep in mind:
        - The session id is an integer, so avoid trying to access it as a dictionary.""",
        "input_parameters": ["diecutter_id:int"],
        "output_values": ["session_id:int"],
        "module": "database"
    }

    def call(diecutter_id: int) -> int:
        # set the random seed
        np.random.seed(GetCurrentSessionIdFromDiecutterId.seed+diecutter_id)
        # generate random session id
        session_id = np.random.randint(0, 100)
        return session_id
