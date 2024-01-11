# Panorama Creator for Videos

`panorama.py` is a Python script that constructs a panoramic image from a video file. It focus on the video taken with the horizontal movement.

## Dependencies

- Python 3.x
- OpenCV (cv2)
- NumPy
- tqdm

## Installation of Dependencies

You can install the required dependencies with the assistance of pip:

```bash
pip install opencv-python numpy tqdm
```

## Usage

Run the `panorama.py` script in your terminal specifying the input video path, and, if desired, an output filename for the panorama:

```bash
python panorama.py <path_to_input_video> [-o <desired_output_panorama_path>]
```

### Parameters:

- `<path_to_input_video>`: The path to the video file from which you want to create the panorama. This is a mandatory parameter.
- `-o <desired_output_panorama_path>`: The path and filename for the panorama image to be saved. This is optional. If not specified, the output will be saved as 'output_panorama.jpg' in the current working directory.

## Example

Below is an example of how to run the script with the input video path as mandatory and the output panorama path as optional:

```bash
python panorama.py myvideo.MOV -o mypanorama.jpg
```

If you choose not to provide an output panorama path, the script will save the panoramic image as `output_panorama.jpg` by default:

```bash
python panorama.py myvideo.MOV
```

## Features

- Robust feature detection and matching using ORB (Oriented FAST and Rotated BRIEF) algorithm.
- Image warping and stitching for seamless panorama construction.
- Customizable sliding window width to handle varying degrees of overlap between frames.
- Automatic handling of termination signals; the script saves the current state of the panorama if interrupted.

## Safety and Cleanup

The script registers signal handlers to ensure that the current progress is not lost even if the process gets an interrupt (`CTRL+C`) or termination signal. A partially completed panorama is saved automatically in response to these signals.

## Considerations

The current implementation assumes that the camera is mostly stationary with primarily horizontal movement. Significant variations in camera position or vertical movement may affect the quality of the resulting panorama.

## Contributions and License

This project is open for contributions. Feel free to fork, modify, and suggest enhancements or fixes through pull requests. The project is provided as-is with no warranty, and use of the code is under your own discretion. 

Happy panoramic image capturing!