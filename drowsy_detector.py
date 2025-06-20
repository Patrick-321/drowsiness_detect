import queue
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import sys
import os
import platform
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QHBoxLayout,
                             QWidget, QVBoxLayout, QFrame, QGraphicsDropShadowEffect,
                             QPushButton, QGridLayout)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot, QPropertyAnimation, QRect
from PyQt5.QtMultimedia import QSound

# Set environment variable for macOS
os.environ['QT_MAC_WANTS_LAYER'] = '1'

# Try to import platform-specific sound libraries
try:
    if platform.system() == 'Windows':
        import winsound
    elif platform.system() == 'Darwin':  # macOS
        import subprocess
except ImportError:
    pass


class VideoThread(QThread):
    """Separate thread for video processing to avoid GUI blocking"""
    changePixmap = pyqtSignal(QImage)
    updateStats = pyqtSignal(dict)
    alertSignal = pyqtSignal(int)  # 0: normal, 1: warning, 2: danger

    def __init__(self):
        super().__init__()
        self.running = True

        # Initialize face mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.points_ids = [187, 411, 152, 68, 174, 399, 298]

        # State variables
        self.yawn_state = ''
        self.left_eye_state = ''
        self.right_eye_state = ''

        self.blinks = 0
        self.microsleeps = 0
        self.yawns = 0
        self.yawn_duration = 0

        self.left_eye_still_closed = False
        self.right_eye_still_closed = False
        self.yawn_in_progress = False

        # Alert tracking
        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds between alerts

        try:
            # Initialize YOLO models
            self.detectyawn = YOLO("model/yawn_detect/best.pt")
            self.detecteye = YOLO("model/eye_detect/best.pt")
        except Exception as e:
            print(f"Warning: Could not load YOLO models: {e}")
            self.detectyawn = None
            self.detecteye = None

    def run(self):
        """Main video processing loop"""
        try:
            self.cap = cv2.VideoCapture(0)

            # Check if camera opened successfully
            if not self.cap.isOpened():
                print("Error: Cannot open camera")
                return

            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    self.process_frame(frame)
                else:
                    print("Error: Cannot read frame")
                    break

        except Exception as e:
            print(f"Error in video thread: {e}")
        finally:
            if hasattr(self, 'cap'):
                self.cap.release()

    def process_frame(self, frame):
        """Process a single frame"""
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    ih, iw, _ = frame.shape
                    points = []

                    for point_id in self.points_ids:
                        lm = face_landmarks.landmark[point_id]
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        points.append((x, y))

                    if len(points) >= 7:
                        # Extract ROIs
                        x1, y1 = points[0]
                        x2, _ = points[1]
                        _, y3 = points[2]
                        x4, y4 = points[3]
                        x5, y5 = points[4]
                        x6, y6 = points[5]
                        x7, y7 = points[6]

                        x6, x7 = min(x6, x7), max(x6, x7)
                        y6, y7 = min(y6, y7), max(y6, y7)

                        # Ensure ROI coordinates are valid
                        if (x2 > x1 and y3 > y1 and x5 > x4 and y5 > y4 and
                                x7 > x6 and y7 > y6):

                            mouth_roi = frame[y1:y3, x1:x2]
                            right_eye_roi = frame[y4:y5, x4:x5]
                            left_eye_roi = frame[y6:y7, x6:x7]

                            if self.detecteye and self.detectyawn:
                                try:
                                    # Predict states
                                    if left_eye_roi.size > 0:
                                        self.left_eye_state = self.predict_eye(left_eye_roi, self.left_eye_state)
                                    if right_eye_roi.size > 0:
                                        self.right_eye_state = self.predict_eye(right_eye_roi, self.right_eye_state)
                                    if mouth_roi.size > 0:
                                        self.predict_yawn(mouth_roi)
                                except Exception as e:
                                    print(f"Error in prediction: {e}")

                            # Update states
                            self.update_states()

                            # Draw annotations
                            self.draw_annotations(frame, points)

            # Convert and emit frame
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.changePixmap.emit(qt_image)

            # Emit stats update
            stats = {
                'blinks': self.blinks,
                'microsleeps': self.microsleeps,
                'yawns': self.yawns,
                'yawn_duration': self.yawn_duration
            }
            self.updateStats.emit(stats)

            # Check for alerts
            current_time = time.time()
            if current_time - self.last_alert_time > self.alert_cooldown:
                if round(self.microsleeps, 2) > 2.0:
                    self.alertSignal.emit(2)  # Danger
                    self.last_alert_time = current_time
                elif round(self.yawn_duration, 2) > 2.0:
                    self.alertSignal.emit(1)  # Warning
                    self.last_alert_time = current_time
                else:
                    self.alertSignal.emit(0)  # Normal

        except Exception as e:
            print(f"Error processing frame: {e}")

    def predict_eye(self, eye_frame, eye_state):
        """Predict eye state"""
        try:
            results_eye = self.detecteye.predict(eye_frame, verbose=False)
            boxes = results_eye[0].boxes
            if boxes is not None and len(boxes) > 0:
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy()
                max_confidence_index = np.argmax(confidences)
                class_id = int(class_ids[max_confidence_index])

                if class_id == 1:
                    eye_state = "Close Eye"
                elif class_id == 0 and confidences[max_confidence_index] > 0.30:
                    eye_state = "Open Eye"
        except Exception as e:
            print(f"Error in eye prediction: {e}")

        return eye_state

    def predict_yawn(self, yawn_frame):
        """Predict yawn state"""
        try:
            results_yawn = self.detectyawn.predict(yawn_frame, verbose=False)
            boxes = results_yawn[0].boxes

            if boxes is not None and len(boxes) > 0:
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy()
                max_confidence_index = np.argmax(confidences)
                class_id = int(class_ids[max_confidence_index])

                if class_id == 0:
                    self.yawn_state = "Yawn"
                elif class_id == 1 and confidences[max_confidence_index] > 0.50:
                    self.yawn_state = "No Yawn"
        except Exception as e:
            print(f"Error in yawn prediction: {e}")

    def update_states(self):
        """Update detection states"""
        # Eyes closed detection
        if self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye":
            if not self.left_eye_still_closed and not self.right_eye_still_closed:
                self.left_eye_still_closed, self.right_eye_still_closed = True, True
                self.blinks += 1
            self.microsleeps += 45 / 1000
        else:
            if self.left_eye_still_closed and self.right_eye_still_closed:
                self.left_eye_still_closed, self.right_eye_still_closed = False, False
            self.microsleeps = 0

        # Yawn detection
        if self.yawn_state == "Yawn":
            if not self.yawn_in_progress:
                self.yawn_in_progress = True
                self.yawns += 1
            self.yawn_duration += 45 / 1000
        else:
            if self.yawn_in_progress:
                self.yawn_in_progress = False
                self.yawn_duration = 0

    def draw_annotations(self, frame, points):
        """Draw face landmarks on frame"""
        if len(points) >= 7:
            # Draw eye regions
            cv2.rectangle(frame, (points[3][0], points[3][1]),
                          (points[4][0], points[4][1]), (0, 255, 0), 2)
            cv2.rectangle(frame, (min(points[5][0], points[6][0]), min(points[5][1], points[6][1])),
                          (max(points[5][0], points[6][0]), max(points[5][1], points[6][1])), (0, 255, 0), 2)

            # Draw mouth region
            cv2.rectangle(frame, (points[0][0], points[0][1]),
                          (points[1][0], points[2][1]), (255, 0, 255), 2)

    def stop(self):
        """Stop the thread"""
        self.running = False
        self.wait()


class DrowsinessDetector(QMainWindow):
    def __init__(self):
        super().__init__()

        # Statistics
        self.stats = {
            'blinks': 0,
            'microsleeps': 0,
            'yawns': 0,
            'yawn_duration': 0
        }

        # Alert state
        self.current_alert_level = 0
        self.flash_timer = QTimer()
        self.flash_timer.timeout.connect(self.flash_alert)
        self.flash_state = False

        # Setup UI
        self.setup_ui()

        # Create and start video thread
        self.video_thread = VideoThread()
        self.video_thread.changePixmap.connect(self.update_image)
        self.video_thread.updateStats.connect(self.update_stats)
        self.video_thread.alertSignal.connect(self.handle_alert)
        self.video_thread.start()

    def setup_ui(self):
        """Setup the UI"""
        self.setWindowTitle("AI Drowsiness Detection System")
        self.setGeometry(100, 100, 1100, 650)

        # Set simple style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
            }
            QFrame {
                background-color: #2d2d2d;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
        """)

        # Central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Main layout
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Left panel - Video
        left_panel = self.create_video_panel()
        main_layout.addWidget(left_panel, 2)

        # Right panel - Stats
        right_panel = self.create_stats_panel()
        main_layout.addWidget(right_panel, 1)

    def create_video_panel(self):
        """Create video panel"""
        panel = QFrame()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Live Video Feed")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("border: 2px solid #0d7377; background-color: #000000;")
        self.video_label.setText("Initializing camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        return panel

    def create_stats_panel(self):
        """Create statistics panel"""
        panel = QFrame()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("Detection Statistics")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Alert label
        self.alert_label = QLabel("System Active")
        self.alert_label.setAlignment(Qt.AlignCenter)
        self.alert_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                padding: 20px;
                background-color: #27ae60;
                border-radius: 10px;
                margin: 10px;
            }
        """)
        layout.addWidget(self.alert_label)

        # Warning frame (initially hidden)
        self.warning_frame = QFrame()
        self.warning_frame.setStyleSheet("""
            QFrame {
                background-color: #e74c3c;
                border: 3px solid #c0392b;
                border-radius: 10px;
                padding: 20px;
                margin: 10px;
            }
        """)
        self.warning_frame.hide()

        warning_layout = QVBoxLayout(self.warning_frame)

        self.warning_icon = QLabel("⚠️")
        self.warning_icon.setAlignment(Qt.AlignCenter)
        self.warning_icon.setStyleSheet("font-size: 48px; background-color: transparent;")
        warning_layout.addWidget(self.warning_icon)

        self.warning_text = QLabel("DROWSINESS DETECTED!")
        self.warning_text.setAlignment(Qt.AlignCenter)
        self.warning_text.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 20px;
                font-weight: bold;
                background-color: transparent;
            }
        """)
        warning_layout.addWidget(self.warning_text)

        layout.addWidget(self.warning_frame)

        # Stats grid
        stats_widget = QWidget()
        stats_grid = QGridLayout(stats_widget)
        stats_grid.setSpacing(15)

        # Create stat labels
        self.blink_label = self.create_stat_widget("👁️ Blinks:", "0")
        self.microsleep_label = self.create_stat_widget("💤 Microsleeps:", "0.0s")
        self.yawn_label = self.create_stat_widget("😮 Yawns:", "0")
        self.yawn_duration_label = self.create_stat_widget("⏱️ Yawn Duration:", "0.0s")

        stats_grid.addWidget(self.blink_label, 0, 0)
        stats_grid.addWidget(self.microsleep_label, 0, 1)
        stats_grid.addWidget(self.yawn_label, 1, 0)
        stats_grid.addWidget(self.yawn_duration_label, 1, 1)

        layout.addWidget(stats_widget)

        # Reset button
        self.reset_button = QPushButton("Reset Stats")
        self.reset_button.clicked.connect(self.reset_stats)
        layout.addWidget(self.reset_button, alignment=Qt.AlignCenter)

        layout.addStretch()

        return panel

    def create_stat_widget(self, label, value):
        """Create a stat widget"""
        widget = QLabel(f"{label} {value}")
        widget.setStyleSheet("""
            QLabel {
                font-size: 16px;
                padding: 15px;
                background-color: #34495e;
                border-radius: 8px;
            }
        """)
        widget.setAlignment(Qt.AlignCenter)
        return widget

    @pyqtSlot(QImage)
    def update_image(self, image):
        """Update video display"""
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(scaled_pixmap)

    @pyqtSlot(dict)
    def update_stats(self, stats):
        """Update statistics display"""
        self.stats = stats

        # Update labels
        self.blink_label.setText(f"👁️ Blinks: {stats['blinks']}")
        self.microsleep_label.setText(f"💤 Microsleeps: {round(stats['microsleeps'], 2)}s")
        self.yawn_label.setText(f"😮 Yawns: {stats['yawns']}")
        self.yawn_duration_label.setText(f"⏱️ Yawn Duration: {round(stats['yawn_duration'], 2)}s")

    @pyqtSlot(int)
    def handle_alert(self, alert_level):
        """Handle alert signals from video thread"""
        self.current_alert_level = alert_level

        if alert_level == 0:  # Normal
            self.flash_timer.stop()
            self.warning_frame.hide()
            self.alert_label.show()
            self.alert_label.setText("✓ System Active")
            self.alert_label.setStyleSheet("""
                QLabel {
                    font-size: 18px;
                    font-weight: bold;
                    padding: 20px;
                    background-color: #27ae60;
                    border-radius: 10px;
                    margin: 10px;
                }
            """)
        elif alert_level == 1:  # Warning - Prolonged Yawn
            self.alert_label.hide()
            self.warning_frame.show()
            self.warning_text.setText("⚠️ PROLONGED YAWN DETECTED!")
            self.warning_frame.setStyleSheet("""
                QFrame {
                    background-color: #e67e22;
                    border: 3px solid #d35400;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px;
                }
            """)
            self.play_alert_sound(1)
            self.flash_timer.start(500)  # Flash every 500ms
        elif alert_level == 2:  # Danger - Drowsiness
            self.alert_label.hide()
            self.warning_frame.show()
            self.warning_text.setText("🚨 DROWSINESS ALERT! 🚨")
            self.warning_frame.setStyleSheet("""
                QFrame {
                    background-color: #e74c3c;
                    border: 3px solid #c0392b;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px;
                }
            """)
            self.play_alert_sound(2)
            self.flash_timer.start(300)  # Flash faster for danger

    def flash_alert(self):
        """Create flashing effect for alerts"""
        if self.flash_state:
            self.warning_frame.setStyleSheet("""
                QFrame {
                    background-color: #2c3e50;
                    border: 3px solid #34495e;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px;
                }
            """)
        else:
            if self.current_alert_level == 1:
                self.warning_frame.setStyleSheet("""
                    QFrame {
                        background-color: #e67e22;
                        border: 3px solid #d35400;
                        border-radius: 10px;
                        padding: 20px;
                        margin: 10px;
                    }
                """)
            else:
                self.warning_frame.setStyleSheet("""
                    QFrame {
                        background-color: #e74c3c;
                        border: 3px solid #c0392b;
                        border-radius: 10px;
                        padding: 20px;
                        margin: 10px;
                    }
                """)
        self.flash_state = not self.flash_state

    def play_alert_sound(self, level):
        """Play alert sound based on level"""
        try:
            system = platform.system()

            if system == 'Windows':
                if level == 1:  # Warning
                    winsound.Beep(800, 500)  # 800Hz for 500ms
                else:  # Danger
                    # Play multiple beeps for danger
                    for _ in range(3):
                        winsound.Beep(1200, 300)  # 1200Hz for 300ms
                        time.sleep(0.1)

            elif system == 'Darwin':  # macOS
                if level == 1:
                    # Use system sound for warning
                    subprocess.run(['afplay', '/System/Library/Sounds/Hero.aiff'])
                else:
                    # Use more urgent sound for danger
                    subprocess.run(['afplay', '/System/Library/Sounds/Sosumi.aiff'])

            elif system == 'Linux':
                # Use system beep command
                if level == 1:
                    os.system('beep -f 800 -l 500')
                else:
                    os.system('beep -f 1200 -l 300 -r 3')

            else:
                # Fallback - try using QSound
                QSound.play("alert.wav")

        except Exception as e:
            print(f"Could not play sound: {e}")
            # If sound fails, at least print a warning
            print("ALERT: Drowsiness detected!")

    def reset_stats(self):
        """Reset statistics"""
        # Reset in video thread
        self.video_thread.blinks = 0
        self.video_thread.microsleeps = 0
        self.video_thread.yawns = 0
        self.video_thread.yawn_duration = 0

    def closeEvent(self, event):
        """Handle window close"""
        self.flash_timer.stop()
        self.video_thread.stop()
        event.accept()


def main():
    # Create QApplication
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show window
    window = DrowsinessDetector()
    window.show()

    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()