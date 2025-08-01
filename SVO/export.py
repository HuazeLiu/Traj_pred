import sys
import os
import cv2
import glob
import numpy as np
import pyzed.sl as sl
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define output directories and files
output_dir = "./data/images/"
image0_file_path = os.path.join(output_dir, "image_0/")
image1_file_path = os.path.join(output_dir, "image_1/")
gt_file_path = os.path.join(output_dir, "groundtruth.tum")
times_file_path = os.path.join(output_dir, "times.txt")

# Make sure folders exist
os.makedirs(image0_file_path, exist_ok=True)
os.makedirs(image1_file_path, exist_ok=True)

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()

def record_img_data(LeftImg, RightImg, Timestamp):
    left_img_save_path = os.path.join(image0_file_path, f"{Timestamp}.png")
    right_img_save_path = os.path.join(image1_file_path, f"{Timestamp}.png")
    cv2.imwrite(left_img_save_path, LeftImg)
    cv2.imwrite(right_img_save_path, RightImg)
    
def record_groundtruth_data( data ):
        timestamp = data[0]
        [tx,ty,tz] = data[1:4:1]
        [qx,qy,qz,qw] = data[4:8:1]
        with open(gt_file_path, 'a') as file_handle:
            file_handle.write("{0} {1} {2} {3} {4} {5} {6} {7}\n".format(timestamp,
                                                                     tx,
                                                                     ty,
                                                                     tz,
                                                                     qx,
                                                                     qy,
                                                                     qz,
                                                                     qw))
            file_handle.close()

def record_timestamp(timestamp):
    with open(times_file_path, 'a') as file_handle:
        file_handle.write(f"{timestamp * 0.000001:.6f}\n")
        file_handle.close


if __name__ == "__main__":
    ## Clear file generated before
    if os.path.isfile(times_file_path):
        os.remove(times_file_path)
        print("Times file Deleted successfully")
    else:
        print("Times file does not exist")
    if( os.path.isfile(gt_file_path)):
        os.remove(gt_file_path)
        print("Groundtruth file Deleted successfully")
    else:
        print("Groundtruth file does not exist")
    for image_file in glob.glob(image0_file_path + '*'):
        if(image_file.__len__() != 0):
            os.remove(image_file)
            print("delete"+str(image_file))
        else:
            print("image_0 is Already empty")
    for image_file in glob.glob(image1_file_path + '*'):
        if(image_file.__len__() != 0):
            os.remove(image_file)
            print("delete"+str(image_file))
        else:
            print("image_1 is Already empty")
            
    ## Setting SVO parameter
    svo_file_path = Path(sys.argv[1])
    camera_params = sl.InitParameters(
        svo_real_time_mode = False,  # Don't convert in realtime
        coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD,
        coordinate_units = sl.UNIT.METER,
        depth_mode = sl.DEPTH_MODE.ULTRA
    )
    camera_params.set_from_svo_file(str(svo_file_path))

    ## Open camera
    zed = sl.Camera()
    camera_status = zed.open(camera_params)
    if camera_status != sl.ERROR_CODE.SUCCESS:
        print("Fail to open camera offline!")
        exit(1)
    print("Camera start to work offline!") 

    print("Total frames in SVO:", zed.get_svo_number_of_frames())
    
    ## Set Tracking params
    track_params = sl.PositionalTrackingParameters()
    track_status = zed.enable_positional_tracking(track_params)
    if track_status != sl.ERROR_CODE.SUCCESS:
        print("Fail to enable tracking!")
        exit(1)
    print("Successfully enable tracking function!")
    
    ## Create series variables
    pose = sl.Pose()                        # Create pose var
    Transform = sl.Transform()              # Create transform var
    Tranlation = sl.Translation()           # Create translation var
    Quaternion = sl.Orientation()           # Create quaternion var
    LeftImage = sl.Mat()                    # Create Left image mat
    RightImage = sl.Mat()                   # Create right image mat
    
    tx_list = []
    ty_list = []
    tz_list = []
    
    ## Export data
    runtime_params = sl.RuntimeParameters()
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")
    nb_frames = zed.get_svo_number_of_frames()      # Get total frames number
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()   # Get current frame id 
            
            # Retrieve stereo images and timestamp
            zed.retrieve_image(LeftImage, sl.VIEW.LEFT)
            zed.retrieve_image(RightImage, sl.VIEW.RIGHT)
            left_img = LeftImage.get_data()
            right_img = RightImage.get_data()            
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_microseconds()

            if left_img is None or left_img.shape[0] == 0:
                print(f"Warning: Empty left image at frame {svo_position}")


            # Retrieve pose 
            pose_status = zed.get_position(
                py_pose = pose,
                reference_frame = sl.REFERENCE_FRAME.WORLD,
            )

            record_img_data(left_img , right_img , timestamp)
            record_timestamp(timestamp)

            if pose_status == sl.POSITIONAL_TRACKING_STATE.OK:
                
                # Get 4x4 Matrix of Transform (ROS frame)
                Transform = pose.pose_data(sl.Transform())
            
                # Get Translation (ROS frame)
                Tranlation = pose.get_translation(sl.Translation())
                tx = Tranlation.get()[0]
                ty = Tranlation.get()[1]
                tz = Tranlation.get()[2]
                
                # Get quaternion express orientation (ROS frame)
                Quaternion = pose.get_orientation(sl.Orientation())
                qx = Quaternion.get()[0]
                qy = Quaternion.get()[1]
                qz = Quaternion.get()[2]
                qw = Quaternion.get()[3]
                
                tx_list.append(tx)
                ty_list.append(ty)
                tz_list.append(tz)
                
                # Record data
                data_wrapper = [timestamp,tx,ty,tz,qx,qy,qz,qw]
                record_groundtruth_data(data_wrapper)
            else:
                print(f"Pose not tracked at timestamp {timestamp}")


                
            # Display progress
            progress_bar((svo_position + 1) / nb_frames * 100, 30)
            
            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break

    fig = plt.figure()    
    ax = fig.add_subplot(111, projection='3d') 
    ax.plot(tx_list,ty_list,tz_list) 
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50) 
    ax.set_zlim(-50, 50)
    ax.legend()
    plt.grid()         
    plt.show()
    ## Close camera    
    zed.close()
    print("Camera stop")
