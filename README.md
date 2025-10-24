# Automated Attendance System with Raspberry Pi and Face Recognition

## About The Project

A Raspberry Pi based automated attendance system that uses face recognition to register students and log attendance in real time. The project provides a Tkinter GUI to capture headshots with the Pi camera, train a recognition model, run live attendance capture, and view logs saved as CSV.

## Built With

- Python 3  
- Tkinter (GUI)  
- OpenCV (face detection and recognition)  
- Picamera2 (Raspberry Pi camera integration)  
- CSV (attendance logs)  
- Threading (non-blocking UI operations)

## Getting Started

These instructions will get you a copy of the project up and running on your local Raspberry Pi for development and testing purposes.

### Prerequisites

- Raspberry Pi with camera module connected and enabled  
- Raspberry Pi OS with Python 3.7 or later  
- Recommended Python packages: opencv-python; picamera2; pillow; numpy  
- Tkinter (usually included with standard Python installs)

### Installation

1. Clone the repo
```bash
git clone https://github.com/goldkrs/MyProj1.git
cd MyProj1
```

2. Create and activate a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Python dependencies
```bash
# If requirements.txt exists
pip install -r requirements.txt

# Or install core packages manually
pip install opencv-python picamera2 pillow numpy
```

### Hardware Setup

- Attach the Raspberry Pi Camera to the CSI connector and enable the camera in raspi-config.  
- Verify camera operation using Picamera2 example scripts before running the GUI.  
- Ensure sufficient storage for captured images; the `dataset/` directory will store headshots per student.

## Usage

Run the GUI:
```bash
python gui.py
```

Primary GUI actions
- Capture Student — Enter USN then capture multiple headshots using the camera; images saved to `dataset/<USN>/image_*.jpg`.  
- Train Photos — Runs the training pipeline that processes images in `dataset/` and prepares the recognition model.  
- Capture Attendance — Starts real time recognition; recognized USNs with timestamps are appended to `entry_log.csv`.  
- Stop Capture — Stops live recognition thread.  
- View Attendance — Opens a window that reads and displays `entry_log.csv`.  
- Exit — Stops any running threads and closes the application.

Command line utilities
```bash
python train_model.py
python rec_attendance.py
python headshots_picam.py
```

## Project Structure

- `gui.py` — Main Tkinter GUI and app launcher  
- `headshots_picam.py` — Camera capture utilities for headshots using Picamera2  
- `predictFaces.py` — Inference helper functions for predicting faces  
- `facial_req.py` — Face detection and helper utilities  
- `rec_attendance.py` — Real time attendance capture and CSV logging  
- `train_model.py` — Training pipeline for face recognition model  
- `dataset/` — Captured face images organized by USN (created at runtime)  
- `entry_log.csv` — CSV attendance log with columns: `usn`, `time`  
- `assets/` — Optional screenshots or demo images (add if available)  
- `requirements.txt` — Python dependency list (create or update if missing)

## Implementation Notes

- Dataset layout must be: `dataset/<USN>/image_0.jpg, image_1.jpg, ...` for training.  
- Attendance log format: CSV with fields `usn` and `time`.  
- GUI uses threads so long running tasks do not block the interface; ensure threads are cleanly stopped before exiting.  
- For headless operation remove or adapt any OpenCV `imshow` usage to save frames without display.  
- If recognition is slow, reduce camera resolution or optimize the model pipeline.

## Roadmap

- Add explicit model persistence and versioned model files  
- Add cloud synchronization with Google Sheets or Firebase for centralized logs  
- Add admin authentication and export options (CSV or PDF)  
- Improve recognition accuracy with augmented data and modern embeddings  
- Build a web dashboard using Flask or FastAPI to view and export attendance remotely

## Contributing

Contributions are welcome.

1. Fork the Project  
2. Create your Feature Branch (git checkout -b feature/AmazingFeature)  
3. Commit your Changes (git commit -m "Add some AmazingFeature")  
4. Push to the Branch (git push origin feature/AmazingFeature)  
5. Open a Pull Request

Update `requirements.txt` when adding dependencies and include screenshots or short demo GIFs under `assets/` for UI changes.


## Contact

Hem — hemkishorp98@gmail.com

Project Link: https://github.com/goldkrs/MyProj1

## Acknowledgments

- Raspberry Pi documentation and Picamera2 examples  
- OpenCV tutorials for face detection and recognition
## License
This project is licensed under the MIT License.  
