import os, sys, cv2, numpy as np, pyzed.sl as sl
from inference.models import YOLOWorld

YOLO_MODEL_ID = "yolo_world/l"
YOLO_CLASSES = ['tennis ball']
SAVE_MP4 = True
OUT_MP4_NAME = "drop_detect_y.mp4"

CONF_THRES = 0.003
DT = 1/60  # 60 FPS
Y_DROP_THRESH = 0.30  # meters/frame

yolo = YOLOWorld(model_id=YOLO_MODEL_ID)
yolo.set_classes(YOLO_CLASSES)

zed = sl.Camera()
init = sl.InitParameters(
    camera_resolution=sl.RESOLUTION.HD720,
    camera_fps=60,
    depth_mode=sl.DEPTH_MODE.ULTRA,
    coordinate_units=sl.UNIT.METER)
if zed.open(init) != sl.ERROR_CODE.SUCCESS:
    print("ZED open failed"); sys.exit(1)

runtime = sl.RuntimeParameters()
left_img = sl.Mat()
pc_mat = sl.Mat()

info = zed.get_camera_information().camera_configuration
fx, fy = info.calibration_parameters.left_cam.fx, info.calibration_parameters.left_cam.fy
cx, cy = info.calibration_parameters.left_cam.cx, info.calibration_parameters.left_cam.cy
h, w = info.resolution.height, info.resolution.width

writer = None
if SAVE_MP4:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_MP4_NAME, fourcc, int(round(1/DT)), (w, h))


prev_Y = None
in_hand = True

print("Press Q to quit.")


while True:
    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        continue

    zed.retrieve_image(left_img, sl.VIEW.LEFT)
    zed.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
    imgL = cv2.cvtColor(left_img.get_data(), cv2.COLOR_BGRA2BGR)

    # Object detection
    result = yolo.infer(imgL, confidence=CONF_THRES)
    if not result.predictions:
        cv2.imshow("drop", imgL)
        if writer: writer.write(imgL)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    best = max(result.predictions, key=lambda p: p.confidence)
    u, v = int(best.x), int(best.y)

    # Query 3D position at (u, v)
    xyz_rgba = pc_mat.get_value(u, v)[1]
    X, Y, Z, _ = xyz_rgba
    if not np.isfinite(Y) or Z == 0:
        cv2.imshow("drop", imgL)
        if writer: writer.write(imgL)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    if in_hand and prev_Y is not None:
        y_vel = (Y - prev_Y) / DT
        if y_vel > Y_DROP_THRESH:
            in_hand = False
    prev_Y = Y

    u_proj = int(fx * X / Z + cx)
    v_proj = int(fy * Y / Z + cy)
    color = (0, 0, 255) if in_hand else (0, 255, 0)
    label = "IN HAND" if in_hand else "DROPPED"

    cv2.circle(imgL, (u_proj, v_proj), 8, color, cv2.FILLED)
    cv2.putText(imgL, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    box_w, box_h = int(best.width), int(best.height)
    xmin = int(best.x - box_w / 2)
    ymin = int(best.y - box_h / 2)
    xmax = int(best.x + box_w / 2)
    ymax = int(best.y + box_h / 2)
    cv2.rectangle(imgL, (xmin, ymin), (xmax, ymax), color, 2)

    cv2.imshow("drop", imgL)
    if writer: writer.write(imgL)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

zed.close()
if writer: writer.release()
cv2.destroyAllWindows()
print("Finished. Video saved to", OUT_MP4_NAME if writer else "N/A")