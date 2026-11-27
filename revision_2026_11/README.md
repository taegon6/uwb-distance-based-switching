# Revised manuscript: code and derived data

This folder contains the analysis code and the derived data of the revised manuscript

> T.-g. Lee, D.-w. Seo, and B.-J. Jang, "Distance-Based Switching of UWB Ranging for Electric Vehicle Wireless Charging Alignment," *Energies* (MDPI), manuscript energies-4704455, revision 1.

The revised manuscript reports 39 vehicle trials recorded on 24 November 2026: 20 on the straight path, 15 on the curved path, and 4 on the sharply curved path.

## Labels in the files

The labels were set in the logging software before the experiment and are kept so that the code runs on the files.

| In the files | In the manuscript |
|---|---|
| S, C1, C2 | straight path, curved path, sharply curved path |
| FIXED_SS, FIXED_DS, FIXED_PHASE | fixed SS-TWR, fixed DS-TWR, fixed high-precision DS-TWR |
| STAGED | proposed method (distance-based switching) |
| SS, DS, PHASE | SS-TWR, DS-TWR, high-precision DS-TWR |
| far, mid | alignment distance from 4 to 10.5 m, from 0.5 to 4 m |

## Files behind the manuscript

Paths of data files are relative to `derived_data/`, and scripts are in `code/`.

| Manuscript | Data | Script |
|---|---|---|
| Table 2 | revision_checks/frame_time_budget.csv | frame_budget.py |
| Table 3, Figure 7a | revision_checks/table3_v2_mean_sd.csv, calculated from camera_eval/segment_errors.csv and stops/stops_v2_seltos_L439.csv | cam_eval.py, table3_mean_sd.py, plot_fig7_v5.py |
| Paired differences of the proposed method from the fixed methods (Section 4.3, rows with variant log) | camera_eval/paired_blocks.csv | cam_eval.py |
| Table 4, Figure 7c,d | revision_checks/stop_decomposition_hp_stops.csv (see below) | hp_fixed.py |
| Table 5 | revision_checks/whole_approach_by_run.csv, whole_approach_frames_by_run.csv, baton_timeouts_by_run.csv | whole_approach.py, whole_approach_frames.py |
| Figure 6 | raw logs | plot_routes_v6.py |
| Figure 7b, Section 4.4 | stops/stops_v2_seltos_L439.csv, stops/stop_implied_link_offsets_v2.csv | analyze_stops_v2.py |
| Figure 8 | revision_checks/sequential_link_model_10p5.csv | motion_skew_plot_v5.py |
| Section 4.1.2 (time alignment, tablet clock, registration) | camera_eval/sync_check.csv, camera_eval/tablet_clock_ticks_v2.csv, revision_checks/tablet_clock_check.csv, camera_eval/registration_*.json | cam_eval.py, tablet_check.py |
| Section 4.2 (validity check) | revision_checks/valid_share_by_phase.csv | valid_share_by_phase.py |
| Section 4.3 (shift of the camera time, link error on the curved path) | revision_checks/sync_sensitivity.csv, revision_checks/range_residual_c1_by_heading.csv | revision_checks.py, range_residual.py |
| Section 4.4 (manual entries, precision of a link distance) | revision_checks/reference_sign_sensitivity.csv, revision_checks/hp_precision_vs_window.csv | gt_sensitivity.py, hp_precision.py |
| Section 4.5 | revision_checks/threshold_composite_10p5.csv, confirmation_frames_replay_v2.csv, alternative_staging_v2.csv | threshold_composite_10p5.py, revision_checks3.py |
| Section 5 (computation time, delay to the PC) | revision_checks/pose_compute_time.csv, camera_eval/dejitter_stats.csv | revision_checks2.py, cam_eval_dejit.py |

In stop_decomposition_hp_stops.csv, the offsets none, laboratory, and field are no calibration, the laboratory calibration, and the field calibration. The aggregations single frame, 25 samples, and 3 s are a single frame, the median over 25 frames, and the median over 3 s. The methods DS and HP fixed are DS-TWR and high-precision DS-TWR processed as in Section 2.2. Rows with the aggregation jitter give the precision. Table 4 gives the mean ± standard deviation of rx_cm over the 16 trials.

All other files in derived_data/ and code/ are earlier versions or additional analyses that the revised manuscript does not report. They are kept as a record of the analysis.

## Notes on the code

- The scripts were run with Python 3.12, NumPy, pandas, and OpenCV. Most of them contain absolute paths of the analysis computer (D:/sandbox/uwb/output/field_20261124) and read the raw run folders, which are not part of this repository. They document the processing as it was run and are not a packaged library. table3_mean_sd.py runs on the files of this repository.
- Frame-level tables larger than 1.5 MB are not included.

## Coordinates

The scripts use a coordinate system that is mirrored with respect to the manuscript: x_paper = -x_code, y_paper = y_code, and heading_paper = -heading_code. The same reflection applies to the body coordinates of the tags. Anchor A1 is at (1, -3) m in the manuscript and at (-1, -3) m in the configuration file. Lateral errors and distances are not affected.

## Raw data

The raw UWB logs and the videos are available from the corresponding author (Byung-Jun Jang, bjjang@kookmin.ac.kr) on request because of their size.

