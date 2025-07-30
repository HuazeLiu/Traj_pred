import os, sys, math, csv, cv2, numpy as np, pandas as pd, pyzed.sl as sl
from inference.models import YOLOWorld
from scipy.optimize import minimize
"""
traj_pred_ego_state.py
----------------------
Egocentric 3D trajectory predictor (static ZED camera) **with full state extraction**
Outputs per frame CSV containing:
    frame,   x, y, z,   vx, vy, vz,   eta   (s),  success_flag
and a second CSV with the predicted future trajectory.
"""
YOLO_MODEL_ID   = "yolo_world/l"
YOLO_CLASSES    = ["wallet"] # edit when needed
CONF_THRES      = 0.003
CAM_FPS         = 30
DT              = 1.0 / CAM_FPS

MIN_FIT_POINTS  = 5  # start fitting after this many pts
PRED_HORIZON    = 15  # future frames predicted each step

SAVE_MP4        = True
OUT_MP4_NAME    = "bottle.mp4"
CSV_STATE_PATH  = "ego_state_log.csv"
CSV_PRED_PATH   = "ego_pred_traj.csv"

# Catch‑zone definition (camera frame)
CATCH_CENTER_CAM = np.array([0.5, 0.0, 0.8])   # m
CATCH_RADIUS     = 0.10  # m, 10 cm

# helping the fit function be concave down
def objective_function(params, t, y):
    a, b, c = params
    return np.sum((y - (a * t**2 + b * t + c))**2)

def main():
    zed = sl.Camera()
    init = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD1080,
                             camera_fps=CAM_FPS,
                             depth_mode=sl.DEPTH_MODE.ULTRA,   # no neural
                             coordinate_units=sl.UNIT.METER)
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("ZED open failed"); sys.exit(1)

    runtime = sl.RuntimeParameters()
    left_img = sl.Mat(); pc_mat = sl.Mat()

    # Intrinsics
    cam_info = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    fx, fy, cx, cy = cam_info.fx, cam_info.fy, cam_info.cx, cam_info.cy

    # Detector
    yolo = YOLOWorld(model_id=YOLO_MODEL_ID); yolo.set_classes(YOLO_CLASSES)
    # Output containers
    state_log, pred_log = [], []
    # Optional video
    writer = None
    if SAVE_MP4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        h, w = cam_info.image_size.height, cam_info.image_size.width
        writer = cv2.VideoWriter(OUT_MP4_NAME, fourcc, CAM_FPS, (w, h))

    all_gts = []
    frame_idx = 0
    print("Press  q  to quit.")

    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        zed.retrieve_image(left_img, sl.VIEW.LEFT)
        zed.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)

        img = cv2.cvtColor(left_img.get_data(), cv2.COLOR_BGRA2BGR)
        result = yolo.infer(img, confidence=CONF_THRES)

        # Ground‑truth extraction
        latest_gt = {"frame": frame_idx, "x": np.nan, "y": np.nan, "z": np.nan}
        if result.predictions:
            best = max(result.predictions, key=lambda p: p.confidence)
            u, v = int(best.x), int(best.y)
            err, xyz_rgba = pc_mat.get_value(u, v)
            X, Y, Z, _ = xyz_rgba
            if np.isfinite(X):
                latest_gt.update({"x": X, "y": Y, "z": Z})
                cv2.rectangle(img, (u-10, v-10), (u+10, v+10), (0,0,255), 2)
                cv2.circle(img, (u, v), 5, (0,255,0), cv2.FILLED)
        all_gts.append(latest_gt)

        #Prediction & state extraction
        valid = [gt for gt in all_gts if not np.isnan(gt['x'])]
        if len(valid) >= MIN_FIT_POINTS:
            history = pd.DataFrame(valid[-10:])
            t_hist = history['frame'].values * DT
            xs, ys, zs = history['x'].values, history['y'].values, history['z'].values

            # Fit: x,z linear ; y quadratic (Ay≥0)
            Bx, Cx = np.polyfit(t_hist, xs, 1)
            Bz, Cz = np.polyfit(t_hist, zs, 1)
            init_guess = np.polyfit(t_hist, ys, 2)
            bounds = [(0, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)]
            Ay, By, Cy = minimize(objective_function, init_guess, args=(t_hist, ys), bounds=bounds).x

            # Current state (position & velocity)
            t_now = frame_idx * DT
            x_now = Bx * t_now + Cx
            y_now = Ay * t_now**2 + By * t_now + Cy
            z_now = Bz * t_now + Cz
            vx, vy, vz = Bx, 2*Ay*t_now + By, Bz

            # Future predictions
            future_frames = np.arange(frame_idx+1, frame_idx+1+PRED_HORIZON)
            t_future = future_frames * DT
            pred_x = Bx * t_future + Cx
            pred_y = Ay * t_future**2 + By*t_future + Cy
            pred_z = Bz * t_future + Cz

            # ETA to catch zone
            eta = np.nan; success=False
            for tf, px, py, pz in zip(t_future, pred_x, pred_y, pred_z):
                if np.linalg.norm(np.array([px,py,pz]) - CATCH_CENTER_CAM) <= CATCH_RADIUS:
                    eta = tf - t_now; success=True; break

            # Log state
            state_log.append({
                'frame': frame_idx,
                'x': x_now, 'y': y_now, 'z': z_now,
                'vx': vx,   'vy': vy,   'vz': vz,
                'eta': eta, 'success': int(success)
            })

            # Store predictions
            for f_idx, px, py, pz in zip(future_frames, pred_x, pred_y, pred_z):
                pred_log.append({'made_at_frame': frame_idx,
                                 'pred_for_frame': int(f_idx),
                                 'x': px, 'y': py, 'z': pz})

            # Visualize future traj
            prev_pt=None
            for px,py,pz in zip(pred_x, pred_y, pred_z):
                if pz<=0: continue
                u_p = int(fx*px/pz + cx); v_p = int(fy*py/pz + cy)
                cv2.circle(img,(u_p,v_p),3,(255,0,255),cv2.FILLED)
                if prev_pt is not None: cv2.line(img, prev_pt,(u_p,v_p),(0,255,255),1)
                prev_pt=(u_p,v_p)
            if success:
                cv2.putText(img, "ETA %.2fs"%eta, (30,100), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        # UI & controls
        cv2.putText(img, f"Frame {frame_idx}", (20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.imshow("ego-traj-predictor", img)
        if writer: writer.write(img)
        if cv2.waitKey(1)&0xFF==ord('q'): break
        frame_idx += 1

    # ───── Save logs ─────
    pd.DataFrame(state_log).to_csv(CSV_STATE_PATH,index=False)
    pd.DataFrame(pred_log).to_csv(CSV_PRED_PATH, index=False)
    print(f"Saved {CSV_STATE_PATH} and {CSV_PRED_PATH}")

    zed.close(); cv2.destroyAllWindows()
    if writer: writer.release()

if __name__ == "__main__":
    main()
