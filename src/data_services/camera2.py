import numpy as np
import time


class GetFrameFromCamera2:
    seed = 0
    delay = 0.05
    refresh_rate = 1
    last_execution_time = 0
    last_value = None
    call_parameters_list = ['camera2_id']
    description = {
        "brief_description": "Data service that, given the id of a camera2, provides a frame captured from that camera2.",
        "detailed_description": 
        """Data service that, given the id of a camera2, provides a frame captured from that camera2.
        Recall that a camera2 is a camera device that points downwards to a conveyor belt of a specific production line that trasports stack of cutout cardboards produced by a specific diecutter.
        The data service takes a single parameter, namely the id of the camera2 (an integer) and returns a frame captured from that camera2 as a numpy matrix.
        The matrix is a 2D array having a shape of (1080, 1920, 3) where 1080 is the height, 1920 is the width and 3 is the number of channels (RGB).

        Example usage:
        - If the id of the camera2 is 123, then the data service would be called as follows:
        camera2_id = 123
        frame = GetFrameFromCamera2.call(camera2_id=123)
        # assuming the frame is a numpy matrix
        print(frame.shape)  # (1080, 1920, 3)

        Things to keep in mind:
        - The refresh rate of the camera is 1 second, i.e. the frame is updated every second, so if the data service is called multiple times within a second, it will return the same value.
        - The frame is a numpy matrix, so avoid trying to access it as a dictionary.""",
        "input_parameters": ["camera2_id:int"],
        "output_values": ["frame:np.matrix"],
        "module": "camera2"
    }

    def call(camera2_id) -> np.matrix:
        # add a small delay to simulate the time it takes to get the frame from the camera
        time.sleep(GetFrameFromCamera2.delay)
        current_execution_time = time.time()
        # check if the refresh rate has passed or if it is the first time the function is called
        if GetFrameFromCamera2.last_value is None or current_execution_time - GetFrameFromCamera2.last_execution_time >= GetFrameFromCamera2.refresh_rate:
            # set the random seed
            np.random.seed(GetFrameFromCamera2.seed+camera2_id)
            # generate a random frame
            frame = np.random.rand(1080, 1920, 3)
            # update the seed
            GetFrameFromCamera2.seed += 1
            # update the last value
            GetFrameFromCamera2.last_value = frame
            # update the last execution time
            GetFrameFromCamera2.last_execution_time = current_execution_time
            return frame
        else:
            return GetFrameFromCamera2.last_value


class DetectErrorsInCardboardStackUsingCamera2:
    seed = 0
    delay = 0.05
    call_parameters_list = ['camera2_id', 'frame']
    description = {
        "brief_description": "Data service that, given a frame captured from a specific camera2 and the identifier of that camera2, detects whether the frame contains a stack of cardboards with errors.",
        "detailed_description":
        """Data service that, given a frame captured from a specific camera2 and the identifier of that camera2, detects whether the frame contains a stack of cardboards with errors.
        Recall that a camera2 is a camera device that points downwards to a conveyor belt of a specific production line that trasports stack of cutout cardboards produced by a specific diecutter.
        It takes two parameters, the id of the camera2 to use and the frame captured from that camera2 that contains a stack of cardboards.
        It returns a boolean value, True if the frame contains a stack of cardboards with errors, False otherwise.

        Example usage:
        - If frame is a variable containing the frame captured from the camera2 with id 123, then the data service would be called as follows:
        camera2_id = 123
        contains_errors, error_type = DetectErrorsInCardboardStack.call(camera2_id=123, frame=frame)

        Things to keep in mind:
        - The output value is a boolean, so avoid trying to access it as a dictionary.""",
        "input_parameters": ["camera2_id:int", "frame:np.matrix"],
        "output_values": ["contains_errors:bool"],
        "module": "camera2"
    }

    def call(camera2_id: int, frame: np.matrix) -> bool:
        # set the random seed
        np.random.seed(DetectErrorsInCardboardStackUsingCamera2.seed+camera2_id)
        # detect if the frame contains a cardboard
        has_errors =  bool(np.random.choice([True, False]))
        if has_errors == False:
            error_type = 0
        else:
            error_type = int(np.random.choice([1,2]))
        # add a small delay to simulate the time it takes to detect the cardboard
        time.sleep(DetectErrorsInCardboardStackUsingCamera2.delay)
        # update the seed
        DetectErrorsInCardboardStackUsingCamera2.seed += 1
        return has_errors, error_type
