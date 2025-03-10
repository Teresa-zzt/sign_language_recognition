import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QGridLayout, QLabel, QAction, QVBoxLayout, QLineEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ImageScrollArea(QScrollArea):
    def __init__(self, folder_path, image_files):
        super().__init__()
        self.folderPath = folder_path
        self.imageFiles = image_files
        self.totalImages = len(self.imageFiles)
        self.loadedImages = 0
        self.imagesPerBatch = 150
        
        self.widget = QWidget()
        self.gridLayout = QGridLayout(self.widget)
        self.setWidget(self.widget)
        self.setWidgetResizable(True)
        
        self.loadNextBatch()

        self.verticalScrollBar().valueChanged.connect(self.onScroll)

    # loads 150 images at a time
    def loadNextBatch(self):
        # Calculates the rows and columns to place the images at
        row, col = divmod(self.loadedImages, 10)
        end_index = min(self.loadedImages + self.imagesPerBatch, self.totalImages)
        # Load the next 150 images
        for i in range(self.loadedImages, end_index):
            pixmap = QPixmap(os.path.join(self.folderPath, self.imageFiles[i]))
            label = QLabel()
            label.setPixmap(pixmap.scaled(60, 60, Qt.KeepAspectRatio))
            self.gridLayout.addWidget(label, row, col)
            col += 1
            # If we have reached the end of the row, move to the next row
            if col == 10:
                col = 0
                row += 1
        self.loadedImages = end_index

    def onScroll(self, value):
        # Check if the scrollbar has reached the bottom
        if value == self.verticalScrollBar().maximum():
            self.loadNextBatch()