#!/usr/bin/env python3
"""
Test script for the optimized drowsiness detector interface.
This script tests the UI components without requiring camera access.
"""

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                             QPushButton, QLabel, QHBoxLayout)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
import random

# Import the custom components from the main file
from drowsy_detector_cleaned import StatusIndicator, AlertBanner


class TestInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness Detector - Interface Test")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("🧪 Interface Test - Drowsiness Detector")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 24px;
                font-weight: bold;
                color: #212529;
                margin-bottom: 20px;
            }
        """)
        layout.addWidget(title)
        
        # Alert banner
        self.alert_banner = AlertBanner()
        layout.addWidget(self.alert_banner)
        
        # Status indicators
        self.blink_indicator = StatusIndicator("Blinks", "👁️")
        self.microsleep_indicator = StatusIndicator("Microsleeps", "💤")
        self.yawn_indicator = StatusIndicator("Yawns", "😮")
        self.yawn_duration_indicator = StatusIndicator("Yawn Duration", "⏳")
        
        # Add indicators to layout
        indicators_layout = QHBoxLayout()
        indicators_layout.addWidget(self.blink_indicator)
        indicators_layout.addWidget(self.microsleep_indicator)
        indicators_layout.addWidget(self.yawn_indicator)
        indicators_layout.addWidget(self.yawn_duration_indicator)
        layout.addLayout(indicators_layout)
        
        # Test buttons
        button_layout = QHBoxLayout()
        
        test_blink_btn = QPushButton("Test Blink Counter")
        test_blink_btn.clicked.connect(self.test_blink_counter)
        test_blink_btn.setStyleSheet("""
            QPushButton {
                background: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #0056b3;
            }
        """)
        
        test_alert_btn = QPushButton("Test Alert Banner")
        test_alert_btn.clicked.connect(self.test_alert_banner)
        test_alert_btn.setStyleSheet("""
            QPushButton {
                background: #dc3545;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #c82333;
            }
        """)
        
        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self.reset_all)
        reset_btn.setStyleSheet("""
            QPushButton {
                background: #6c757d;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #545b62;
            }
        """)
        
        button_layout.addWidget(test_blink_btn)
        button_layout.addWidget(test_alert_btn)
        button_layout.addWidget(reset_btn)
        layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("Interface test ready. Click buttons to test components.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
                color: #6c757d;
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
                margin-top: 20px;
            }
        """)
        layout.addWidget(self.status_label)
        
        # Initialize values
        self.blink_count = 0
        self.microsleep_count = 0.0
        self.yawn_count = 0
        self.yawn_duration = 0.0
        
        # Timer for simulating real-time updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulate_updates)
        self.timer.start(1000)  # Update every second
        
    def test_blink_counter(self):
        """Simulate blink detection"""
        self.blink_count += random.randint(1, 3)
        self.blink_indicator.update_value(self.blink_count, 50)
        self.status_label.setText(f"Blink counter updated: {self.blink_count} blinks detected")
        
    def test_alert_banner(self):
        """Test alert banner functionality"""
        alert_types = ["warning", "danger"]
        alert_messages = [
            "Prolonged Yawn Detected!",
            "Microsleep Risk Detected!",
            "Drowsiness Alert!"
        ]
        
        alert_type = random.choice(alert_types)
        message = random.choice(alert_messages)
        self.alert_banner.show_alert(message, alert_type)
        self.status_label.setText(f"Alert banner shown: {message}")
        
    def reset_all(self):
        """Reset all indicators and hide alert"""
        self.blink_count = 0
        self.microsleep_count = 0.0
        self.yawn_count = 0
        self.yawn_duration = 0.0
        
        self.blink_indicator.update_value(0, 50)
        self.microsleep_indicator.update_value(0.0, 10)
        self.yawn_indicator.update_value(0, 20)
        self.yawn_duration_indicator.update_value(0.0, 10)
        
        self.alert_banner.hide()
        self.status_label.setText("All indicators reset to zero")
        
    def simulate_updates(self):
        """Simulate real-time updates for testing"""
        # Randomly update microsleep duration
        if random.random() < 0.3:  # 30% chance
            self.microsleep_count += random.uniform(0.1, 0.5)
            self.microsleep_indicator.update_value(round(self.microsleep_count, 2), 10)
            
        # Randomly update yawn duration
        if random.random() < 0.2:  # 20% chance
            self.yawn_duration += random.uniform(0.1, 0.3)
            self.yawn_duration_indicator.update_value(round(self.yawn_duration, 2), 10)
            
        # Randomly increment yawn count
        if random.random() < 0.1:  # 10% chance
            self.yawn_count += 1
            self.yawn_indicator.update_value(self.yawn_count, 20)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application-wide stylesheet
    app.setStyleSheet("""
        QApplication {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
    """)
    
    window = TestInterface()
    window.show()
    sys.exit(app.exec_()) 