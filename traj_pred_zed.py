import os, sys, math, cv2, numpy as np, pyzed.sl as sl
from tqdm import tqdm
import supervision as sv
from inference.models import YOLOWorld

YOLO_MODEL_ID   = "yolo_world/l"
YOLO_CLASSES    = ['tennis ball']
SAVE_MP4        = True
OUT_MP4_NAME    = "ball_traj_live.mp4"

MIN_FIT_POINTS  = 5 # start fitting after this many 3D points
FIT_WINDOW      = 5   # equally-spaced samples used for polyfit
PRED_HORIZON    = 15 # frames to predict
CONF_THRES      = 0.003 # Yolo confidence score
DT              = 1/60   # 60 Hz capture


yolo = YOLOWorld(model_id="yolo_world/l")
yolo.set_classes(YOLO_CLASSES)
zed = sl.Camera()
init = sl.InitParameters(camera_resolution = sl.RESOLUTION.HD720,
                         camera_fps        = 60,
                         depth_mode        = sl.DEPTH_MODE.ULTRA,
                         coordinate_units  = sl.UNIT.METER)
if zed.open(init) != sl.ERROR_CODE.SUCCESS:
    print("ZED open failed"); sys.exit(1)

runtime  = sl.RuntimeParameters()
left_img = sl.Mat()
pc_mat   = sl.Mat()  # XYZRGBA point cloud

h, w = zed.get_camera_information().camera_configuration.resolution.height, \
       zed.get_camera_information().camera_configuration.resolution.width

# video writer
writer = None
if SAVE_MP4:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_MP4_NAME, fourcc, int(round(1/DT)), (w, h))


xs, ys, zs, ts = [], [], [], []
frame_idx      = 0
print("Press  q  in the display window to quit.")

while True:
    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        continue
    zed.retrieve_image(left_img, sl.VIEW.LEFT)
    zed.retrieve_measure(pc_mat,  sl.MEASURE.XYZRGBA)

    imgL = cv2.cvtColor(left_img.get_data(), cv2.COLOR_BGRA2BGR)
    intrinsics = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy

    result = yolo.infer(imgL, confidence=CONF_THRES)
    if result.predictions:
        best = max(result.predictions, key=lambda p: p.confidence)
        
        #Draw the current detection
        x, y, w, h = int(best.x), int(best.y), int(best.width), int(best.height)
        xmin, ymin = x - w // 2, y - h // 2
        xmax, ymax = x + w // 2, y + h // 2
        cv2.rectangle(imgL, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(imgL, best.class_name, (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.circle(imgL, (x, y), 8, (0, 255, 0), cv2.FILLED)

        #Get 3D point and add it to history
        err, xyz_rgba = pc_mat.get_value(x, y)
        X, Y, Z, _ = xyz_rgba
        if np.isfinite(X):
            xs.append(X); ys.append(Y); zs.append(Z); ts.append(frame_idx)


    if xs: # Only draw if we have points
        x_draw, y_draw, z_draw = xs[-25:], ys[-25:], zs[-25:]
        for Xp, Yp, Zp in zip(x_draw, y_draw, z_draw):
            if not np.isfinite(Xp) or Zp <= 0: continue
            u_tr = int(fx * Xp / Zp + cx)
            v_tr = int(fy * Yp / Zp + cy)
            cv2.circle(imgL, (u_tr, v_tr), 3, (0, 255, 0), cv2.FILLED)

    # Fit the model and draw the predicted future trajectory
    if len(xs) >= MIN_FIT_POINTS:
        t_newest, x_newest, y_newest, z_newest = ts[-10:], xs[-10:], ys[-10:], zs[-10:]
        
        k = min(FIT_WINDOW, len(x_newest))
        idx_sample = np.linspace(0, len(x_newest) - 1, k, dtype=int)
        
        t_s  = np.array(t_newest)[idx_sample]
        xs_s = np.array(x_newest)[idx_sample]
        ys_s = np.array(y_newest)[idx_sample]
        zs_s = np.array(z_newest)[idx_sample]

        Bx, Cx = np.polyfit(t_s, xs_s, 1)      # Linear fit for horizontal axis
        Ay, By, Cy = np.polyfit(t_s, ys_s, 2)  # Quadratic fit for vertical axis
        Bz, Cz = np.polyfit(t_s, zs_s, 1)      # Linear fit for depth axis
        
        future_t = np.arange(frame_idx + 1, frame_idx + 1 + PRED_HORIZON)
        pred_x = Bx * future_t + Cx
        pred_y = Ay * future_t**2 + By * future_t + Cy
        pred_z = Bz * future_t + Cz

        for Xp, Yp, Zp in zip(pred_x, pred_y, pred_z):
            if not np.isfinite(Zp) or Zp <= 0: continue
            u_p = int(fx * Xp / Zp + cx)
            v_p = int(fy * Yp / Zp + cy)
            cv2.circle(imgL, (u_p, v_p), 5, (255, 0, 255), cv2.FILLED)

    cv2.imshow("trajectory", imgL)
    if writer: writer.write(imgL)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_idx += 1


zed.close()
if writer: writer.release()
cv2.destroyAllWindows()
print("Finished. Video saved to", OUT_MP4_NAME if writer else "N/A")
