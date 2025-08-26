import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os
import uuid
import time


class GaitAnalysis:
    def __init__(self, video_path, height, weight, distance, model_path="./model/pose_landmarker_heavy.task"):
        self.video_path = video_path
        self.height = height
        self.weight = weight
        self.distance = distance
        self.model_path = model_path
        self.pose_landmarker_options = self.initialize_landmarker()
        self.frame_rate = None
        self.df = pd.DataFrame()
        self.distance_df = pd.DataFrame()

    def initialize_landmarker(self):
        """Initializes the MediaPipe Pose Landmarker."""
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    def calculate_angle_3d(a, b, c):
        """Calculates the angle between three 3D points (in degrees)."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        dot_product = np.dot(ba, bc)
        norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
        cosine_angle = np.clip(dot_product / (norm_ba * norm_bc), -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    @staticmethod
    def draw_landmarks_on_image(rgb_image, detection_result):
        """Draws pose landmarks and connections on an image."""
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return annotated_image

    @staticmethod
    def save_annotated_video(frames, frame_rate):
        """Saves annotated frames as a VP80 encoded WEBM video."""
        output_directory = "output_videos"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        output_video_filename = f"{uuid.uuid4().hex}.webm"
        output_video_path = os.path.join(output_directory, output_video_filename)

        if not frames:
            return None

        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        return output_video_path

    @staticmethod
    def gap_fill(data):
        """Fills gaps in the data using cubic spline interpolation."""
        x = np.arange(len(data))
        good_indices = np.where(np.isfinite(data))[0]
        if len(good_indices) < 4:
            return np.full_like(data, np.nanmean(data))
        interp_func = interp1d(x[good_indices], data[good_indices], kind='cubic', fill_value="extrapolate")
        return interp_func(x)

    @staticmethod
    def butterworth_low_pass_filter(data, cutoff, fs, order=4):
        """Applies a Butterworth low-pass filter."""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def calculate_accurate_step_count(self, heel_strikes_left, heel_strikes_right, frame_rate,
                                      foot_positions_x, dist_left_filtered, dist_right_filtered):
        """
        Multi-method step counting algorithm that uses:
        1. Peak-based heel strike detection
        2. Foot position velocity analysis
        3. Stride pattern analysis
        4. Cross-validation between methods
        """

        # Method 1: Peak-based detection (existing method, but less restrictive)
        def count_from_peaks():
            heel_strikes_left_time = heel_strikes_left / frame_rate
            heel_strikes_right_time = heel_strikes_right / frame_rate

            combined_heel_strikes = []
            for time_stamp in heel_strikes_left_time:
                combined_heel_strikes.append((time_stamp, 'left'))
            for time_stamp in heel_strikes_right_time:
                combined_heel_strikes.append((time_stamp, 'right'))

            combined_heel_strikes.sort(key=lambda x: x[0])

            if len(combined_heel_strikes) == 0:
                return 0

            # Less restrictive filtering - allow minimum 0.25 seconds between steps
            filtered_steps = []
            last_valid_time = -float('inf')

            for time_stamp, leg in combined_heel_strikes:
                if time_stamp - last_valid_time >= 0.25:  # More lenient
                    filtered_steps.append((time_stamp, leg))
                    last_valid_time = time_stamp

            return len(filtered_steps)

        # Method 2: Foot velocity analysis
        def count_from_foot_velocity():
            left_positions = np.array(foot_positions_x['left'])
            right_positions = np.array(foot_positions_x['right'])

            # Fill gaps
            left_positions_filled = self.gap_fill(left_positions)
            right_positions_filled = self.gap_fill(right_positions)

            # Calculate velocities (change in position)
            left_velocity = np.diff(left_positions_filled)
            right_velocity = np.diff(right_positions_filled)

            # Smooth velocities
            left_velocity_smooth = self.butterworth_low_pass_filter(left_velocity, 3, frame_rate)
            right_velocity_smooth = self.butterworth_low_pass_filter(right_velocity, 3, frame_rate)

            # Find significant forward movements (steps)
            # Use adaptive thresholding based on velocity standard deviation
            left_threshold = np.std(left_velocity_smooth) * 0.3
            right_threshold = np.std(right_velocity_smooth) * 0.3

            # Find peaks in forward velocity (positive for forward movement)
            left_steps, _ = find_peaks(left_velocity_smooth, height=left_threshold, distance=frame_rate * 0.3)
            right_steps, _ = find_peaks(right_velocity_smooth, height=right_threshold, distance=frame_rate * 0.3)

            # Also check for negative peaks (backward relative motion can indicate steps)
            left_steps_neg, _ = find_peaks(-left_velocity_smooth, height=left_threshold, distance=frame_rate * 0.3)
            right_steps_neg, _ = find_peaks(-right_velocity_smooth, height=right_threshold, distance=frame_rate * 0.3)

            # Combine forward and backward motion peaks
            total_left_steps = len(left_steps) + len(left_steps_neg)
            total_right_steps = len(right_steps) + len(right_steps_neg)

            return total_left_steps + total_right_steps

        # Method 3: Stride analysis using both distance and position
        def count_from_stride_analysis():
            # Analyze the hip-ankle distance for more robust step detection
            left_dist = dist_left_filtered
            right_dist = dist_right_filtered

            # Find ALL local maxima and minima as potential step events
            left_maxima, _ = find_peaks(left_dist, prominence=np.std(left_dist) * 0.15, distance=frame_rate * 0.25)
            left_minima, _ = find_peaks(-left_dist, prominence=np.std(left_dist) * 0.15, distance=frame_rate * 0.25)
            right_maxima, _ = find_peaks(right_dist, prominence=np.std(right_dist) * 0.15, distance=frame_rate * 0.25)
            right_minima, _ = find_peaks(-right_dist, prominence=np.std(right_dist) * 0.15, distance=frame_rate * 0.25)

            # Combine all events
            left_events = np.concatenate([left_maxima, left_minima])
            right_events = np.concatenate([right_maxima, right_minima])

            # Each gait cycle typically has 2 major events (heel strike and toe off)
            # So divide by 2 to get approximate step count per leg
            left_steps = len(left_events) // 2
            right_steps = len(right_events) // 2

            return left_steps + right_steps

        # Method 4: Position-based step detection
        def count_from_position_changes():
            left_positions = np.array(foot_positions_x['left'])
            right_positions = np.array(foot_positions_x['right'])

            # Fill gaps
            left_positions_filled = self.gap_fill(left_positions)
            right_positions_filled = self.gap_fill(right_positions)

            # Calculate cumulative displacement
            left_displacement = np.abs(np.diff(left_positions_filled))
            right_displacement = np.abs(np.diff(right_positions_filled))

            # Find significant movements
            left_movement_threshold = np.percentile(left_displacement, 70)  # Top 30% of movements
            right_movement_threshold = np.percentile(right_displacement, 70)

            # Count significant movements as potential steps
            left_significant_moves = np.sum(left_displacement > left_movement_threshold)
            right_significant_moves = np.sum(right_displacement > right_movement_threshold)

            # Each step typically involves 2-3 significant position changes
            # So divide by 2.5 to get approximate step count
            estimated_steps = (left_significant_moves + right_significant_moves) / 2.5

            return int(round(estimated_steps))

        # Calculate using all methods
        method1_count = count_from_peaks()
        method2_count = count_from_foot_velocity()
        method3_count = count_from_stride_analysis()
        method4_count = count_from_position_changes()

        # Combine results using weighted average and validation
        counts = [method1_count, method2_count, method3_count, method4_count]

        # Remove outliers (counts that are more than 2 standard deviations away)
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        filtered_counts = [c for c in counts if abs(c - mean_count) <= 2 * std_count]

        if len(filtered_counts) == 0:
            filtered_counts = counts

        # Use the median of filtered counts for robustness
        final_count = int(round(np.median(filtered_counts)))

        # Additional validation: check if count makes sense given video duration
        total_frames = len(dist_left_filtered)
        total_time = total_frames / frame_rate

        if total_time > 0:
            implied_cadence = (final_count / total_time) * 60  # steps per minute

            # Normal walking cadence is 90-140 steps/minute
            # If too low, might be undercounting; if too high, might be overcounting
            if implied_cadence < 60:  # Too low, likely undercounting
                # Use the higher estimate
                final_count = max(filtered_counts)
            elif implied_cadence > 160:  # Too high, likely overcounting
                # Use the lower estimate
                final_count = min(filtered_counts)

        return max(1, final_count)  # Ensure at least 1 step is counted

    def process_video(self):
        """Processes the video to perform gait analysis."""
        with self.pose_landmarker_options as landmarker:
            cap = cv2.VideoCapture(self.video_path)
            self.frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            frame_number = 0
            annotated_frames = []
            dist_left, dist_right = [], []
            foot_positions_x = {'left': [], 'right': []}

            start_time = time.time()
            frame_count_for_fps = 0
            dynamic_fps = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                h, w, _ = frame.shape

                numpy_frame_from_opencv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
                frame_timestamp_ms = int(frame_number * (1000 / self.frame_rate))
                pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                annotated_image = self.draw_landmarks_on_image(cv2.cvtColor(numpy_frame_from_opencv, cv2.COLOR_RGB2BGR),
                                                               pose_landmarker_result)

                frame_count_for_fps += 1
                if (time.time() - start_time) >= 1.0:
                    dynamic_fps = frame_count_for_fps / (time.time() - start_time)
                    frame_count_for_fps = 0
                    start_time = time.time()
                fps_text = f"Processing FPS: {dynamic_fps:.2f}"
                cv2.putText(annotated_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                left_knee_angle, right_knee_angle, step_width = 0.0, 0.0, 0.0

                if pose_landmarker_result.pose_world_landmarks:
                    world_landmarks = pose_landmarker_result.pose_world_landmarks[0]

                    left_hip = [world_landmarks[23].x, world_landmarks[23].y, world_landmarks[23].z]
                    left_knee = [world_landmarks[25].x, world_landmarks[25].y, world_landmarks[25].z]
                    left_ankle = [world_landmarks[27].x, world_landmarks[27].y, world_landmarks[27].z]
                    right_hip = [world_landmarks[24].x, world_landmarks[24].y, world_landmarks[24].z]
                    right_knee = [world_landmarks[26].x, world_landmarks[26].y, world_landmarks[26].z]
                    right_ankle = [world_landmarks[28].x, world_landmarks[28].y, world_landmarks[28].z]

                    left_knee_angle = self.calculate_angle_3d(left_hip, left_knee, left_ankle)
                    right_knee_angle = self.calculate_angle_3d(right_hip, right_knee, right_ankle)

                    left_foot_x = world_landmarks[31].x
                    right_foot_x = world_landmarks[32].x
                    step_width = abs(left_foot_x - right_foot_x)

                    dist_left.append(np.linalg.norm(np.array(left_hip) - np.array(left_ankle)))
                    dist_right.append(np.linalg.norm(np.array(right_hip) - np.array(right_ankle)))
                    foot_positions_x['left'].append(world_landmarks[31].x)
                    foot_positions_x['right'].append(world_landmarks[32].x)
                else:
                    dist_left.append(np.nan)
                    dist_right.append(np.nan)
                    foot_positions_x['left'].append(np.nan)
                    foot_positions_x['right'].append(np.nan)

                cv2.putText(annotated_image, f"L Knee Angle: {left_knee_angle:.1f} deg", (w - 320, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 191, 0), 2)
                cv2.putText(annotated_image, f"R Knee Angle: {right_knee_angle:.1f} deg", (w - 320, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 191, 0), 2)
                cv2.putText(annotated_image, f"Step Width: {step_width:.2f} m", (w - 320, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 191, 0), 2)

                annotated_frames.append(annotated_image)
                frame_number += 1
            cap.release()

            # --- GAIT ANALYSIS CALCULATIONS ---
            dist_left_filled = self.gap_fill(np.array(dist_left))
            dist_right_filled = self.gap_fill(np.array(dist_right))

            cutoff_freq = 6
            dist_left_filtered = self.butterworth_low_pass_filter(dist_left_filled, cutoff_freq, self.frame_rate)
            dist_right_filtered = self.butterworth_low_pass_filter(dist_right_filled, cutoff_freq, self.frame_rate)

            heel_strikes_left, _ = find_peaks(dist_left_filtered, prominence=np.std(dist_left_filtered) * 0.2,
                                              distance=self.frame_rate * 0.4)
            heel_strikes_right, _ = find_peaks(dist_right_filtered, prominence=np.std(dist_right_filtered) * 0.2,
                                               distance=self.frame_rate * 0.4)

            # Enhanced step counting using the new multi-method algorithm
            step_count = self.calculate_accurate_step_count(heel_strikes_left, heel_strikes_right, self.frame_rate,
                                                            foot_positions_x, dist_left_filtered, dist_right_filtered)

            toe_offs_left, _ = find_peaks(-dist_left_filtered, prominence=np.std(dist_left_filtered) * 0.2,
                                          distance=self.frame_rate * 0.4)
            toe_offs_right, _ = find_peaks(-dist_right_filtered, prominence=np.std(dist_right_filtered) * 0.2,
                                           distance=self.frame_rate * 0.4)

            gait_cycles_left = np.diff(heel_strikes_left) / self.frame_rate
            gait_cycles_right = np.diff(heel_strikes_right) / self.frame_rate

            # --- STATIC TEXT (SUMMARY STATS) ---
            total_frames = frame_number
            if self.frame_rate > 0 and total_frames > 0:
                total_time_seconds = total_frames / self.frame_rate
                try:
                    walked_distance_m = float(self.distance)
                    speed_mps = walked_distance_m / total_time_seconds if total_time_seconds > 0 else 0

                    # Calculate additional metrics
                    cadence = (step_count / total_time_seconds) * 60 if total_time_seconds > 0 else 0
                    avg_step_length = walked_distance_m / step_count if step_count > 0 else 0

                    static_stats_texts = [
                        f"Overall Speed: {speed_mps:.2f} m/s",
                        f"Total Steps: {step_count}",
                        f"Cadence: {cadence:.1f} steps/min",
                        f"Avg Step Length: {avg_step_length:.2f} m",
                        f"Height: {self.height} cm",
                        f"Weight: {self.weight} kg",
                        f"Distance: {self.distance} m",
                        f"Video FPS: {self.frame_rate}"
                    ]
                    for i in range(len(annotated_frames)):
                        y_pos = 60
                        for text in static_stats_texts:
                            cv2.putText(annotated_frames[i], text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (0, 255, 0), 2)
                            y_pos += 30
                except (ValueError, TypeError):
                    print("Warning: Could not display static stats. Ensure input parameters are valid numbers.")

            def calculate_phases(gait_cycles):
                stance = gait_cycles * 0.60
                swing = gait_cycles * 0.40
                loading_response = gait_cycles * 0.10
                mid_stance = gait_cycles * 0.20
                terminal_stance = gait_cycles * 0.20
                pre_swing = gait_cycles * 0.10
                initial_swing = gait_cycles * 0.13
                mid_swing = gait_cycles * 0.14
                terminal_swing = gait_cycles * 0.13
                return stance, swing, loading_response, mid_stance, terminal_stance, pre_swing, initial_swing, mid_swing, terminal_swing

            stance_l, swing_l, load_l, midst_l, termst_l, presw_l, inisw_l, midsw_l, termsw_l = calculate_phases(
                gait_cycles_left)
            stance_r, swing_r, load_r, midst_r, termst_r, presw_r, inisw_r, midsw_r, termsw_r = calculate_phases(
                gait_cycles_right)

            max_len = max(len(gait_cycles_left), len(gait_cycles_right))

            def pad_list(lst, length):
                return np.pad(lst, (0, length - len(lst)), 'constant', constant_values=np.nan)

            self.df = pd.DataFrame({
                'Gait Cycle Left (s)': pad_list(gait_cycles_left, max_len),
                'Stance Left (s)': pad_list(stance_l, max_len),
                'Swing Left (s)': pad_list(swing_l, max_len),
                'Loading Response Left (s)': pad_list(load_l, max_len),
                'Mid Stance Left (s)': pad_list(midst_l, max_len),
                'Terminal Stance Left (s)': pad_list(termst_l, max_len),
                'Pre Swing Left (s)': pad_list(presw_l, max_len),
                'Initial Swing Left (s)': pad_list(inisw_l, max_len),
                'Mid Swing Left (s)': pad_list(midsw_l, max_len),
                'Terminal Swing Left (s)': pad_list(termsw_l, max_len),
                'Gait Cycle Right (s)': pad_list(gait_cycles_right, max_len),
                'Stance Right (s)': pad_list(stance_r, max_len),
                'Swing Right (s)': pad_list(swing_r, max_len),
                'Loading Response Right (s)': pad_list(load_r, max_len),
                'Mid Stance Right (s)': pad_list(midst_r, max_len),
                'Terminal Stance Right (s)': pad_list(termst_r, max_len),
                'Pre Swing Right (s)': pad_list(presw_r, max_len),
                'Initial Swing Right (s)': pad_list(inisw_r, max_len),
                'Mid Swing Right (s)': pad_list(midsw_r, max_len),
                'Terminal Swing Right (s)': pad_list(termsw_r, max_len),
            })

            self.df = self.df.round(4)

            foot_pos_left_filled = self.gap_fill(np.array(foot_positions_x['left']))
            foot_pos_right_filled = self.gap_fill(np.array(foot_positions_x['right']))

            def calculate_phase_distances(heel_strikes, toe_offs, foot_positions, phase_durations):
                cycle_dist, stance_dist, swing_dist, load_dist, midst_dist, termst_dist, presw_dist, inisw_dist, midsw_dist, termsw_dist = (
                    [] for _ in range(10))
                for i in range(len(heel_strikes) - 1):
                    start_cycle = heel_strikes[i]
                    end_cycle = heel_strikes[i + 1]
                    current_toe_offs = toe_offs[(toe_offs > start_cycle) & (toe_offs < end_cycle)]
                    if len(current_toe_offs) == 0: continue
                    toe_off_event = current_toe_offs[0]
                    cycle_dist.append(abs(foot_positions[end_cycle] - foot_positions[start_cycle]))
                    stance_dist.append(abs(foot_positions[toe_off_event] - foot_positions[start_cycle]))
                    swing_dist.append(abs(foot_positions[end_cycle] - foot_positions[toe_off_event]))
                    load_frames = int(phase_durations['load'][i] * self.frame_rate)
                    midst_frames = int(phase_durations['midst'][i] * self.frame_rate)
                    presw_frames = int(phase_durations['presw'][i] * self.frame_rate)
                    inisw_frames = int(phase_durations['inisw'][i] * self.frame_rate)
                    midsw_frames = int(phase_durations['midsw'][i] * self.frame_rate)
                    load_dist.append(abs(foot_positions[start_cycle + load_frames] - foot_positions[start_cycle]))
                    midst_dist.append(abs(foot_positions[start_cycle + load_frames + midst_frames] - foot_positions[
                        start_cycle + load_frames]))
                    termst_dist.append(
                        abs(foot_positions[toe_off_event] - foot_positions[start_cycle + load_frames + midst_frames]))
                    presw_dist.append(abs(foot_positions[toe_off_event + presw_frames] - foot_positions[toe_off_event]))
                    inisw_dist.append(abs(foot_positions[toe_off_event + presw_frames + inisw_frames] - foot_positions[
                        toe_off_event + presw_frames]))
                    midsw_dist.append(abs(
                        foot_positions[toe_off_event + presw_frames + inisw_frames + midsw_frames] - foot_positions[
                            toe_off_event + presw_frames + inisw_frames]))
                    termsw_dist.append(abs(foot_positions[end_cycle] - foot_positions[
                        toe_off_event + presw_frames + inisw_frames + midsw_frames]))
                return cycle_dist, stance_dist, swing_dist, load_dist, midst_dist, termst_dist, presw_dist, inisw_dist, midsw_dist, termsw_dist

            phase_durations_l = {'load': load_l, 'midst': midst_l, 'termst': termst_l, 'presw': presw_l,
                                 'inisw': inisw_l, 'midsw': midsw_l, 'termsw': termsw_l}
            phase_durations_r = {'load': load_r, 'midst': midst_r, 'termst': termst_r, 'presw': presw_r,
                                 'inisw': inisw_r, 'midsw': midsw_r, 'termsw': termsw_r}
            (cycle_dist_l, stance_dist_l, swing_dist_l, load_dist_l, midst_dist_l, termst_dist_l, presw_dist_l,
             inisw_dist_l, midsw_dist_l, termsw_dist_l) = calculate_phase_distances(heel_strikes_left, toe_offs_left,
                                                                                    foot_pos_left_filled,
                                                                                    phase_durations_l)
            (cycle_dist_r, stance_dist_r, swing_dist_r, load_dist_r, midst_dist_r, termst_dist_r, presw_dist_r,
             inisw_dist_r, midsw_dist_r, termsw_dist_r) = calculate_phase_distances(heel_strikes_right, toe_offs_right,
                                                                                    foot_pos_right_filled,
                                                                                    phase_durations_r)

            self.distance_df = pd.DataFrame({
                'Gait Cycle Left (m)': pad_list(cycle_dist_l, max_len),
                'Stance Left (m)': pad_list(stance_dist_l, max_len),
                'Swing Left (m)': pad_list(swing_dist_l, max_len),
                'Loading Response Left (m)': pad_list(load_dist_l, max_len),
                'Mid Stance Left (m)': pad_list(midst_dist_l, max_len),
                'Terminal Stance Left (m)': pad_list(termst_dist_l, max_len),
                'Pre Swing Left (m)': pad_list(presw_dist_l, max_len),
                'Initial Swing Left (m)': pad_list(inisw_dist_l, max_len),
                'Mid Swing Left (m)': pad_list(midsw_dist_l, max_len),
                'Terminal Swing Left (m)': pad_list(termsw_dist_l, max_len),
                'Gait Cycle Right (m)': pad_list(cycle_dist_r, max_len),
                'Stance Right (m)': pad_list(stance_dist_r, max_len),
                'Swing Right (m)': pad_list(swing_dist_r, max_len),
                'Loading Response Right (m)': pad_list(load_dist_r, max_len),
                'Mid Stance Right (m)': pad_list(midst_dist_r, max_len),
                'Terminal Stance Right (m)': pad_list(termst_dist_r, max_len),
                'Pre Swing Right (m)': pad_list(presw_dist_r, max_len),
                'Initial Swing Right (m)': pad_list(inisw_dist_r, max_len),
                'Mid Swing Right (m)': pad_list(midsw_dist_r, max_len),
                'Terminal Swing Right (m)': pad_list(termsw_dist_r, max_len),
            })

            self.distance_df = self.distance_df.round(4)

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            y_axis_label = "Distance (m)"
            axs[0].plot(dist_left_filtered, label="Left Leg Hip-Foot Distance", color="dodgerblue")
            axs[0].plot(heel_strikes_left, dist_left_filtered[heel_strikes_left], "o", color="red", markersize=8,
                        label="Heel Strike")
            axs[0].plot(toe_offs_left, dist_left_filtered[toe_offs_left], "x", color="limegreen", markersize=8,
                        markeredgewidth=2, label="Toe-Off")
            axs[0].set_title("Left Leg Gait Analysis", fontsize=16)
            axs[0].set_ylabel(y_axis_label, fontsize=12)
            axs[0].legend(fontsize=10)
            axs[1].plot(dist_right_filtered, label="Right Leg Hip-Foot Distance", color="darkorange")
            axs[1].plot(heel_strikes_right, dist_right_filtered[heel_strikes_right], "o", color="red", markersize=8,
                        label="Heel Strike")
            axs[1].plot(toe_offs_right, dist_right_filtered[toe_offs_right], "x", color="limegreen", markersize=8,
                        markeredgewidth=2, label="Toe-Off")
            axs[1].set_title("Right Leg Gait Analysis", fontsize=16)
            axs[1].set_xlabel("Frame Number", fontsize=12)
            axs[1].set_ylabel(y_axis_label, fontsize=12)
            axs[1].legend(fontsize=10)
            plt.tight_layout()

            output_video_path = self.save_annotated_video(annotated_frames, self.frame_rate)
            result_summary = self.df.mean().to_json(indent=4)

            return output_video_path, self.df, result_summary, plt, self.distance_df, speed_mps, step_count