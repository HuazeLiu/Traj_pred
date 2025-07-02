import os, sys, math, cv2, numpy as np, pyzed.sl as sl
from tqdm import tqdm
import supervision as sv
from inference.models import YOLOWorld
import csv
from scipy.optimize import minimize

def objective_function(params, t, y):
    a, b, c = params
    y_fit = a * t**2 + b * t + c
    error = np.sum((y - y_fit)**2)
    return error

YOLO_MODEL_ID   = "yolo_world/l"
YOLO_CLASSES    = ['bottle']
SAVE_MP4        = True
OUT_MP4_NAME    = "bottle_traj_live.mp4"
OUT_CSV_NAME    = 'bottle_traj_.csv'

MIN_FIT_POINTS  = 5       # start fitting after this many 3-D points
FIT_WINDOW      = 4     # equally-spaced samples used for polyfit
PRED_HORIZON    = 15      
CONF_THRES      = 0.003   # Yolo confidence score
DT              = 1/30 # 30 Hz 


yolo = YOLOWorld(model_id="yolo_world/l")
yolo.set_classes(YOLO_CLASSES)
zed = sl.Camera()
init = sl.InitParameters(camera_resolution = sl.RESOLUTION.HD1080,
                         camera_fps        = 30,
                         depth_mode        = sl.DEPTH_MODE.ULTRA,
                         coordinate_units  = sl.UNIT.METER)
if zed.open(init) != sl.ERROR_CODE.SUCCESS:
    print("ZED open failed"); sys.exit(1)

runtime  = sl.RuntimeParameters()
left_img = sl.Mat()
pc_mat   = sl.Mat()

h, w = zed.get_camera_information().camera_configuration.resolution.height, \
       zed.get_camera_information().camera_configuration.resolution.width

writer = None
if SAVE_MP4:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_MP4_NAME, fourcc, int(round(1/DT)), (w, h))


xs, ys, zs, ts = [], [], [], []
frame_idx       = 0
print("Press  q  in the display window to quit.")

gt = []
pred = []
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
        
        x, y, w_box, h_box = int(best.x), int(best.y), int(best.width), int(best.height)
        xmin, ymin = x - w_box // 2, y - h_box // 2
        xmax, ymax = x + w_box // 2, y + h_box // 2
        cv2.rectangle(imgL, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        cv2.putText(imgL, best.class_name, (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.circle(imgL, (x, y), 8, (0, 255, 0), cv2.FILLED)

        err, xyz_rgba = pc_mat.get_value(x, y)
        X, Y, Z, _ = xyz_rgba
        if np.isfinite(X):
            xs.append(X); ys.append(Y); zs.append(Z); ts.append(frame_idx)
        gt.append((X, Y, Z)) 

    if xs:
        x_draw, y_draw, z_draw = xs[-25:], ys[-25:], zs[-25:]
        for Xp, Yp, Zp in zip(x_draw, y_draw, z_draw):
            if not np.isfinite(Xp) or Zp <= 0: continue
            u_tr = int(fx * Xp / Zp + cx)
            v_tr = int(fy * Yp / Zp + cy)
            cv2.circle(imgL, (u_tr, v_tr), 3, (0, 255, 0), cv2.FILLED)

    if len(xs) >= MIN_FIT_POINTS:
        t_newest, x_newest, y_newest, z_newest = ts[-10:], xs[-10:], ys[-10:], zs[-10:]
        
        k = min(FIT_WINDOW, len(x_newest))
        idx_sample = np.linspace(0, len(x_newest) - 1, k, dtype=int)
        
        t_s  = np.array(t_newest)[idx_sample]
        xs_s = np.array(x_newest)[idx_sample]
        ys_s = np.array(y_newest)[idx_sample]
        zs_s = np.array(z_newest)[idx_sample]


        initial_guess = np.polyfit(t_s, ys_s, 2)
        y_bounds = [(0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)]
        
        result = minimize(
            objective_function,
            x0=initial_guess,
            args=(t_s, ys_s),
            bounds=y_bounds
        )
        Ay, By, Cy = result.x
        # linear
        Bx, Cx = np.polyfit(t_s, xs_s, 1)
        Bz, Cz = np.polyfit(t_s, zs_s, 1)
        
        future_t = np.arange(frame_idx + 1, frame_idx + 1 + PRED_HORIZON)
        pred_x = Bx * future_t + Cx
        pred_y = Ay * future_t**2 + By * future_t + Cy
        pred_z = Bz * future_t + Cz

        pred.append(list(zip(pred_x, pred_y, pred_z)))

        for Xp, Yp, Zp in zip(pred_x, pred_y, pred_z):
            if not np.isfinite(Zp) or Zp <= 0: continue
            u_p = int(fx * Xp / Zp + cx)
            v_p = int(fy * Yp / Zp + cy)
            cv2.circle(imgL, (u_p, v_p), 5, (255, 0, 255), cv2.FILLED)
            
        curve_t = np.arange(t_s[0], frame_idx + PRED_HORIZON)
        curve_x = Bx * curve_t + Cx
        curve_y = Ay * curve_t**2 + By * curve_t + Cy
        curve_z = Bz * curve_t + Cz
        
        prev_point_2d = None
        for Xp, Yp, Zp in zip(curve_x, curve_y, curve_z):
            if not np.isfinite(Zp) or Zp <= 0: continue
            u_p = int(fx * Xp / Zp + cx)
            v_p = int(fy * Yp / Zp + cy)
            curr_point_2d = (u_p, v_p)
            
            if prev_point_2d is not None:
                cv2.line(imgL, prev_point_2d, curr_point_2d, (0, 255, 255), 2)
            prev_point_2d = curr_point_2d

    cv2.imshow("trajectory", imgL)
    if writer: writer.write(imgL)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_idx += 1

with open(OUT_CSV_NAME, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(['Frame', 'Type', 'X', 'Y', 'Z'])

    for f, (x, y, z) in enumerate(zip(xs, ys, zs)):
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            csv_writer.writerow([ts[f], 'gt', x, y, z])

    for f, triplets in enumerate(pred):
        for (x, y, z) in triplets:
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                pred_frame = ts[-1] + f + 1
                csv_writer.writerow([pred_frame, 'pred', x, y, z])

print("Trajectory saved to", OUT_CSV_NAME)
zed.close()
if writer: writer.release()
cv2.destroyAllWindows()
print("Finished. Video saved to", OUT_MP4_NAME if writer else "N/A")