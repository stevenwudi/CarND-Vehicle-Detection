
# Advanced  Lane Finding

The goals / steps of this project are the following:

* 
*
*

## Implementation details


### 1. Camera Calibration

The goal of camera calibration is distortion correction. Image distortion occurs when a camera looks at 3D objects
in the real world through lens and transform them into a 2D image.
The distortions in an image can:
* Change the apparent size of an object.
* Change the apparent shape of an object.
* Cause an object's appearance to change depending on where it is in the field of view (FOV).
* Make objects appear closer or father away than they actually are.

Cameras use curved lenses to form an image, and light rays often bend a little too much or too little 
at the edges of these lenses. This creates an effect that distorts the edges of images, so that lines or objects
appear more or less curved than they actually are. This is called **radial distortion**, and it is the most common type
of distortion. Another type of distortion, is **tangential distortion**. This occurs when a camera's lens is not aligned
perfectly parallel to the image plane, where the camera film or sensor is. This makes an image look tilted
so that some objects appear father away or closer than they actually are. To correct for these
two types of distortions, a camera calibration is needed (usually via chessboard images).

This is done in `main.py` (by calling `lib/cameraCal.py`) as:
```python
camCal = CameraCal('camera_cal', 'camera_cal/calibrationdata.p')
```
### 2. Image Filters

The imageFilter module will apply filters to an image to isolate the pixels that makes the lane lines. 
The details steps are given as follows:

