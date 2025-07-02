# test_open.py
import pyzed.sl as sl

print("ZED SDK Version:", sl.Camera().get_sdk_version())


init = sl.InitParameters()
cam = sl.Camera()
status = cam.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print("Failed to open camera:", status)
else:
    print("Camera opened successfully")
    cam.close()
