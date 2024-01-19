# For use with YOLOv8
# https://docs.ultralytics.com/modes/predict/#plotting-results

import math
import numpy as np
import cv2
import torch
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ultralytics.data.augment import LetterBox

# https://dl.djicdn.com/downloads/Mavic_2/Mavic_2_Pro_Zoom_User_Manual_v2.2_en.pdf
#fov = 83 # degrees
# The following thread explains why the focal length is so weird.. UGH
# https://forum.dji.com/forum.php?mod=viewthread&tid=215213
focal_length = 4.27 # millimeters
image_width = 3840
image_height = 2160

altitude = 10 # meters
angle = math.radians(30)
video_path = "../uav/thermal.mov"

# mask / bounding_box
area_detection_method = "bounding_box"
draw_bounding_boxes = False
draw_bounding_box_labels = False
draw_masks = False
draw_polygons = False

# sensor_width found through experimentation
sensor_width = 7.638 #6.17
# DO NOT TRUST HEIGHT OR DIAGONAL, these are measured for 4:3 ratio images I believe.
# The videos taken by the drone are in 16:9, it is likely that the sensor-height is
# being clipped when capturing the video.
sensor_height = 4.55
sensor_diagonal = 7.70

#https://www.omnicalculator.com/other/camera-field-of-view
# horizontal_aov found through experimentation
horizontal_aov = 83.619 #71.7#72.83 # degrees
vertical_aov = 57.65
diagonal_aov = 84.08

horizontal_fov = 2 * math.tan(math.radians(horizontal_aov / 2)) * altitude
vertical_fov = 2 * math.tan(math.radians(vertical_aov / 2)) * altitude
diagonal_fov = 2 * math.tan(math.radians(diagonal_aov / 2)) * altitude

# Magic number estimated by manual measurements
pixelToMeterRatio = None#0.0045

yolov8_width = 640
yolov8_height = 384

#focal_length = (image_width/2) / math.tan(math.radians(fov/2)) # pixels

# Takes in point p in the form of (x, y) and matrix is a 3x3 homography matrix
# https://stackoverflow.com/questions/57399915/how-do-i-determine-the-locations-of-the-points-after-perspective-transform-in-t
def findPointInWarpedImage(p, matrix):
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    return (int(px), int(py))

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def modified_matrices_calculate_range_output_without_translation(height, width, overhead_hmatrix,
                                                                    verbose=False):
    range_u = np.array([np.inf, -np.inf])
    range_v = np.array([np.inf, -np.inf])

    i = 0
    j = 0
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    out_upperpixel = v
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])
    i = height - 1
    j = 0
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    out_lowerpixel = v
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])
    i = 0
    j = width - 1
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])
    i = height - 1
    j = width - 1
    u, v, w = np.dot(overhead_hmatrix, [j, i, 1])
    u = u / w
    v = v / w
    if verbose:
        print(u, v)
    range_u[0] = min(u, range_u[0])
    range_v[0] = min(v, range_v[0])
    range_u[1] = max(u, range_u[1])
    range_v[1] = max(v, range_v[1])

    range_u = np.array(range_u, dtype=int)
    range_v = np.array(range_v, dtype=int)

    # it means that while transforming, after some bottom lower image was transformed,
    # upper output pixels got greater than lower
    if out_upperpixel > out_lowerpixel:

        # range_v needs to be updated
        max_height = height * 3
        upper_range = out_lowerpixel
        best_lower = upper_range  # since out_lowerpixel was lower value than out_upperpixel
        #                           i.e. above in image than out_lowerpixel
        x_best_lower = np.inf
        x_best_upper = -np.inf

        for steps_h in range(2, height):
            temp = np.dot(overhead_hmatrix, np.vstack(
                (np.arange(0, width), np.ones((1, width)) * (height - steps_h), np.ones((1, width)))))
            temp = temp / temp[2, :]

            lower_range = temp.min(axis=1)[1]
            x_lower_range = temp.min(axis=1)[0]
            x_upper_range = temp.max(axis=1)[0]
            if x_lower_range < x_best_lower:
                x_best_lower = x_lower_range
            if x_upper_range > x_best_upper:
                x_best_upper = x_upper_range

            if (upper_range - lower_range) > max_height:  # enforcing max_height of destination image
                lower_range = upper_range - max_height
                break
            if lower_range > upper_range:
                lower_range = best_lower
                break
            if lower_range < best_lower:
                best_lower = lower_range
            if verbose:
                print(steps_h, lower_range, x_best_lower, x_best_upper)
        range_v = np.array([lower_range, upper_range], dtype=int)

        # for testing
        range_u = np.array([x_best_lower, x_best_upper], dtype=int)

    return range_u, range_v

def get_overhead_hmatrix_from_4cameraparams(fx, fy, my_tilt, my_roll, img_dims, verbose=False):
    width, height = img_dims

    origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    K3x3 = np.array([[fx, 0, width / 2],
                        [0, fy, height / 2],
                        [0, 0, 1]])

    inv_K3x3 = np.linalg.inv(K3x3)
    if verbose:
        print("K3x3:\n", K3x3)

    R_overhead = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    if verbose:
        print("R_overhead:\n", R_overhead)

    R_slant = rotation_matrix((math.pi / 2) + my_tilt, xaxis)[:3, :3]
    if verbose:
        print("R_slant:\n", R_slant)

    R_roll = rotation_matrix(my_roll, zaxis)[:3, :3]

    middle_rotation = np.dot(R_overhead, np.dot(np.linalg.inv(R_slant), R_roll))

    overhead_hmatrix = np.dot(K3x3, np.dot(middle_rotation, inv_K3x3))
    est_range_u, est_range_v = modified_matrices_calculate_range_output_without_translation(height, width,
                                                                                            overhead_hmatrix,
                                                                                            verbose=False)

    if verbose:
        print("Estimated destination range: u=", est_range_u, "v=", est_range_v)
        print("Altitude: ", altitude)

    moveup_camera = np.array([[1, 0, -est_range_u[0] * 1], [0, 1, -est_range_v[0] * 1], [0, 0, 1]])
    #moveup_camera = np.array([[1, 0, altitude], [0, 1, altitude], [0, 0, 1]])
    if verbose:
        print("moveup_camera:\n", moveup_camera)

    overhead_hmatrix = np.dot(moveup_camera, np.dot(K3x3, np.dot(middle_rotation, inv_K3x3)))
    if verbose:
        print("overhead_hmatrix:\n", overhead_hmatrix)

    return overhead_hmatrix

#homography_matrix = get_overhead_hmatrix_from_4cameraparams(focal_length, focal_length, angle, 0, [image_width, image_height], verbose=True)
#f = (focal_length / sensor_width) * image_width
#f = 0.55897 * image_width
#f = 0.69205 * image_width
f = 0.80205 * image_width
#f = (image_width * 0.5) / math.tan(math.radians(horizontal_aov) * 0.5)
homography_matrix = get_overhead_hmatrix_from_4cameraparams(f, f, angle, 0, [image_width, image_height], verbose=True)

# Load a model
model = YOLO("runs/segment/yolov8Potholes10/weights/best.pt")  # load a custom model

# Open the video file
cap = cv2.VideoCapture(video_path)

# Obtain frame size information using get() method
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width, frame_height)
fps = int(cap.get(5))

output = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

area_estimation_time = 0
yolo_time = 0
video_time = 0

start_time = time.time()

# Loop through the video frames
frames = 0
while cap.isOpened():
    # Read a frame from the video
    start_time_2 = time.time()
    success, frame = cap.read()
    video_time += time.time() - start_time_2

    #for i in range(287):
    #    success, frame = cap.read()

    #warped_img = cv2.warpPerspective(frame, homography_matrix, (image_width*8, image_height*8))
    #warped_img = cv2.warpPerspective(frame[500:,:,:], homography_matrix, (image_width*8, image_height*8))
    #cv2.imwrite("output.png", warped_img)
    #exit(0)

    if success:
        frames += 1
    
        if pixelToMeterRatio is None:
            # Estimate the pixel-to-meter ratio first
            # This should ideally be done once at the beginning of the video. The pixel-to-meter ratio
            # should NOT be changing throughout the video, we're assuming the camera angle is static.
            
            # Cast a ray from the camera to the plane in a straight line following the camera angle
            # The distance between the camera and this "origin" point can be found with a triangle
            origin_distance_in_meters = ( altitude / math.sin(angle) )
            print("origin_distance_in_meters:", origin_distance_in_meters)
            
            # Next, "rotate" the camera by 5 degrees and cast another ray to the plane. Determine the
            # distance between the origin point and this new "rotated" point
            rotation_degrees = 5
            distance_in_meters = ( origin_distance_in_meters / math.sin( math.radians(90 - rotation_degrees) ) ) * math.sin( math.radians(rotation_degrees) )
            print("distance_in_meters:", distance_in_meters)
            
            # Where are these points on the image? The origin should be easy, it's right in the middle
            # of our image. The rotated point can be found using the image width and angle of view
            origin_position_in_pixels = (image_width / 2, image_height / 2)
            rotated_position_in_pixels = ( (image_width / 2) + (image_width / horizontal_aov) * rotation_degrees, image_height / 2 )
            print("origin_position_in_pixels:", origin_position_in_pixels)
            print("rotated_position_in_pixels:", rotated_position_in_pixels)
            
            # Let's warp these points to the top-down image and determine their distance in pixels
            origin_warped_position = findPointInWarpedImage(origin_position_in_pixels, homography_matrix)
            rotated_warped_position = findPointInWarpedImage(rotated_position_in_pixels, homography_matrix)
            print("origin_warped_position:", origin_warped_position)
            print("rotated_warped_position:", rotated_warped_position)
            
            distance_in_pixels = ( (origin_warped_position[0] - rotated_warped_position[0]) ** 2 + (origin_warped_position[1] - rotated_warped_position[1]) ** 2 ) ** 0.5
            print("distance_in_pixels:", distance_in_pixels)
            
            # Now we know the pixel-to-meter ratio!
            pixelToMeterRatio = distance_in_meters / distance_in_pixels
            print("pixelToMeterRatio:", pixelToMeterRatio)
            
            #warpedFrame = cv2.warpPerspective(frame, homography_matrix, (3840*8, 2160*8))
            #cv2.imwrite("big_adjusted.jpg", warpedFrame)
            #exit(0)
    
        # Run YOLOv8 inference on the frame
        start_time_2 = time.time()
        results = model(frame)
        yolo_time += time.time() - start_time_2

        polygon_points = []

        for result in results:
            annotator = Annotator(frame, line_width=4)
            
            boxes = result.boxes
            
            if result.masks and draw_masks:
                #print(result.masks)
                
                pred_masks = result.masks.data
                pred_boxes = boxes
                
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                img_gpu = torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device).permute(
                    2, 0, 1).flip(0).contiguous() / 255
                idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
                annotator.masks(pred_masks.data, colors=[(54, 76, 255)], im_gpu=img_gpu)
            
            for i, box in enumerate(boxes):
                b = box.xyxy[0] # get box coordinates in (left, top, right, bottom) format
                c = box.cls
                
                # Copy the tensor to memory to do CPU processing
                bb = b.cpu()
                
                left, top, right, bottom = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                
                # Estimate the upperbound of the area of the pothole by finding the area of the
                #   polygon that is formed from the four corners of the warped bounding box
                
                areaInMeters = 0
                
                if area_detection_method == "bounding_box":
                    startTime = time.time()
                    
                    # https://stackoverflow.com/a/65141602
                    # topleft -> topright -> bottomright -> bottomleft
                    points = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
                    #print(points)
                    if draw_polygons:
                        polygon_points += list(points)
                    
                    # Convert all points to warped points
                    points = np.array([findPointInWarpedImage(point, homography_matrix) for point in points])
                    #print(points)
                    
                    areaInPixels = cv2.contourArea(points)
                    #p1 = findPointInWarpedImage([left, top], homography_matrix)
                    #p2 = findPointInWarpedImage([right, bottom], homography_matrix)
                    #areaInPixels = abs((p1[0] - p2[0]) * (p1[1] - p2[1]))
                    #print("areaInPixels: ", areaInPixels)
                    areaInMeters = areaInPixels * (pixelToMeterRatio ** 2)
                    #print("areaInMeters: ", areaInMeters)
                    #exit(0)
                    #if draw_polygons:
                    #    polygon_points += [[left, top], [right, bottom]]
                    
                    area_estimation_time += time.time() - startTime
                elif area_detection_method == "mask":
                    startTime = time.time()
                
                    mask = result.masks.data[i]
                    
                    #print("top:",top)
                    #print("bottom:",bottom)
                    #print("left:",left)
                    #print("right:",right)
                    #if bottom - top <= 1 or right - left <= 1:
                    #    continue
                    #elif top < 0 or left < 0 or bottom > image_height or right > image_width:
                    #    continue
                    
                    #mask_image = cv2.cvtColor(mask.cpu().numpy() * 255, cv2.COLOR_BGR2GRAY)
                    # Add a 3rd dimension to the array to make it compatible as a cv2 grayscale image
                    mask_image = np.expand_dims(mask.cpu().numpy() * 255, axis=2).astype("uint8")
                    #_, mask_image = cv2.threshold(mask_image, 200, 255, cv2.THRESH_BINARY)
                    #print(mask_image.dtype)
                    contours, _ = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # contours is an array of this form: [ [[x1, y1]], [[x2, y2]], ... [[xn, yn]] ]
                    #print(contours)
                    #exit(0)
                    
                    if len(contours) == 0:
                        # cv2.findContours could find no contours - mask is probably empty
                        area_estimation_time += time.time() - startTime
                        continue
                       
                    #cv2.imwrite("frame.png", frame)
                    #cv2.imwrite("maskimage.png", mask_image)
                       
                    #print("top:",top)
                    #print("bottom:",bottom)
                    #print("left:",left)
                    #print("right:",right)   
                    #print("before:", contours[0])
                    
                    # Stolen from ultralytics/yolo/utils/ops.py
                    # This code is necessary because there is padding in the mask image that prevents
                    # the points from being linearly transformed with a scaling factor
                    im1_shape = mask_image.shape
                    im0_shape = frame.shape

                    # calculate from im0_shape
                    gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
                    pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding

                    top2, left2 = int(pad[1]), int(pad[0])  # y, x
                    #print(f"pad: top={top2} left={left2}")
                    #bottom2, right2 = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])
                    unpadded_width = yolov8_width - (left2 * 2)
                    unpadded_height = yolov8_height - (top2 * 2)
                    #print(f"pad: unpadded_width={unpadded_width} unpadded_height={unpadded_height}")
                    
                    #contours = [[point[0][0] * (image_width / yolov8_width), point[0][1] * (image_height / yolov8_height)] for point in contours[0]]
                    contours = [[ (point[0][0] - left2) * (image_width / unpadded_width), (point[0][1] - top2) * (image_height / unpadded_height)] for point in contours[0]]
                    
                    #print("after:", contours)
                    #exit(0)
                    
                    if draw_polygons:
                        polygon_points += contours
                    
                    contours = np.array( [[findPointInWarpedImage(point, homography_matrix)] for point in contours] )
                    #print(contours)
                        
                    areaInPixels = cv2.contourArea(contours)
                    #print("areaInPixels", areaInPixels)
                    #areaInPixels = areaInPixels * (image_width / yolov8_width) ** 2
                    areaInMeters = areaInPixels * (pixelToMeterRatio ** 2)
                    #print("areaInMeters", areaInMeters)
                    
                    #img = np.zeros((image_height, image_width))
                    #img = cv2.resize(mask.cpu().numpy() * 255, (image_width, image_height))
                    #warped_img = cv2.warpPerspective(img, homography_matrix, (image_width*8, image_height*8))
                    #warped_img = cv2.warpPerspective(frame, homography_matrix, (image_width*8, image_height*8))
                    #cv2.imwrite("output.png", warped_img)
                    #exit(0)
                    
                    if bottom - top > 200 and right - left > 200:
                        #cv2.imwrite("mask.png", mask.cpu().numpy() * 255)
                        #cv2.imwrite("orginal.png", img)
                        #cv2.imwrite("output.png", warped_img)
                        #exit(0)
                        pass
                        
                    area_estimation_time += time.time() - startTime
                
                #annotator.box_label(b, f"{model.names[int(c)]} conf={box.conf[0]:.2f} area={areaInMeters:.2f}m^2", color=(54, 76, 255))
                if draw_bounding_boxes:
                    annotator.box_label(b, f"{model.names[int(c)]} area={areaInMeters:.2f}m^2" if draw_bounding_box_labels else "", color=(54, 76, 255))
            
        frame = annotator.result()

        for point in polygon_points:
            cv2.circle(frame, ( int(point[0]), int(point[1]) ), 4, (255, 0, 0), -1)

        #warpedFrame = cv2.warpPerspective(frame, homography_matrix, (3840*8, 2160*8))
        #cv2.imwrite("big_adjusted.jpg", warpedFrame)
        #exit(0)

        start_time_2 = time.time()
        output.write(frame)
        video_time += time.time() - start_time_2
        
           
    else:
        # Break the loop if the end of the video is reached
        break

total_time = time.time() - start_time

# Release the video capture object and close the display window
cap.release()
output.release()
cv2.destroyAllWindows()

print(f"total time spent: {total_time:.5f}s")
print(f"frames processed {frames}")
print(f"time spent on area estimation: {area_estimation_time:.5f}s")
print(f"time spent on YOLOv8: {yolo_time:.5f}s")
print(f"time spent on reading/writing video frames: {video_time:.5f}s")

area_estimation_overhead_per_frame = area_estimation_time / frames

print(f"overhead from area estimation: {area_estimation_overhead_per_frame:.5f}s/frame")
