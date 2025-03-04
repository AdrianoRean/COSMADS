import numpy as np
import time


class GetFrameFromCamera1:
    seed = 0
    delay = 0.05
    refresh_rate = 1
    last_execution_time = 0
    last_value = None
    call_parameters_list = ['camera1_id']
    description = {
        "brief_description": "Data service that, given the id of a camera1, provides a frame captured from that camera1.",
        "detailed_description": 
        """Data service that, given the id of a camera1, provides a frame captured from that camera1.
        In general instances of camera1 point downwards to a conveyor belt of a specific production line that trasports single cutout cardboards produced by a specific diecutter.
        The data service takes a single parameter, namely the id of the camera1 (an integer) and returns a frame captured from that camera1 as a numpy matrix.
        The matrix is a 2D array having a shape of (1080, 1920, 3) where 1080 is the height, 1920 is the width and 3 is the number of channels (RGB).

        Example usage:
        - If the id of the camera1 is 123, then the data service would be called as follows:
        camera1_id = 123
        frame = GetFrameFromCamera1.call(camera1_id=123)
        # assuming the frame is a numpy matrix
        print(frame.shape)  # (1080, 1920, 3)

        Things to keep in mind:
        - The refresh rate of the camera is 1 second, i.e. the frame is updated every second, so if the data service is called multiple times within a second, it will return the same value.
        - The frame is a numpy matrix, so avoid trying to access it as a dictionary.""",
        "input_parameters": ["camera1_id:int"],
        "output_values": ["frame:np.matrix"],
        "module": "camera1"
    }

    def call(camera1_id) -> np.matrix:
        # add a small delay to simulate the time it takes to get the frame from the camera
        time.sleep(GetFrameFromCamera1.delay)
        current_execution_time = time.time()
        # check if the refresh rate has passed or if it is the first time the function is called
        if GetFrameFromCamera1.last_value is None or current_execution_time - GetFrameFromCamera1.last_execution_time >= GetFrameFromCamera1.refresh_rate:
            # set the random seed
            np.random.seed(GetFrameFromCamera1.seed+camera1_id)
            # generate a random frame
            frame = np.random.rand(1080, 1920, 3)
            # update the seed
            GetFrameFromCamera1.seed += 1
            # update the last value
            GetFrameFromCamera1.last_value = frame
            # update the last execution time
            GetFrameFromCamera1.last_execution_time = current_execution_time
            return frame
        else:
            return GetFrameFromCamera1.last_value

class DetectErrorsInCardboardUsingCamera1:
    seed = 3
    delay = 0.05
    call_parameters_list = ['camera1_id', 'frame']
    description = {
        "brief_description": "Data service that, given a frame captured from a specific camera1 and the identifier of that camera1, detects whether the frame contains a cardboard with errors and provide the type of error.",
        "detailed_description":
        """Data service that, given a frame captured from a specific camera1 and the identifier of that camera1, detects whether the frame contains a cardboard with errors.
        Recall that a camera1 is a camera device that points downwards to a conveyor belt of a specific production line that trasports single cutout cardboards produced by a specific diecutter.
        It takes two parameters, the id of the camera1 to use and the frame captured from that camera1.
        It returns a boolean value, True if the frame contains a cardboard with errors, False otherwise.
        Also it returns a integer value, 0 if the frame does not contain errors, 1 if the errors are of type hole and 2 if the errors are of type fold. 

        Example usage:
        - If frame is a variable containing the frame captured from the camera1 with id 123, then the data service would be called as follows:
        camera1_id = 123
        contains_errors, type_errors = DetectErrorsInCardboardUsingCamera1.call(camera1_id=123, frame=frame)

        Things to keep in mind:
        - The output is a tuple of boolean and integer, so avoid trying to access it as a dictionary.""",
        "input_parameters": ["camera1_id:int", "frame:np.matrix"],
        "output_values": ["contains_errors:bool", "error_type:int"],
        "module": "camera1"
    }

    def call(camera1_id: int, frame: np.matrix) -> bool:
        # set the random seed
        np.random.seed(DetectErrorsInCardboardUsingCamera1.seed+camera1_id)
        # detect if the frame contains a cardboard
        has_errors =  bool(np.random.choice([True, False]))
        if has_errors == False:
            error_type = 0
        else:
            error_type = int(np.random.choice([1,2]))
        # add a small delay to simulate the time it takes to detect the cardboard
        time.sleep(DetectErrorsInCardboardUsingCamera1.delay)
        # update the seed
        DetectErrorsInCardboardUsingCamera1.seed += 1
        return has_errors, error_type
