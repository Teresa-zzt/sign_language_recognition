import os
import time
from PyQt5.QtCore import pyqtSignal, QObject, QTimer, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QDialog

class Signals(QObject):
    # initalise all signalthat are going to be emitted
    startTimerSignal = pyqtSignal()
    stopTimerSignal = pyqtSignal()
    showGraphSignal = pyqtSignal()
    showTrainingFinishedPopUp = pyqtSignal()

    def __init__(self, save_prefix, label, parent=None):
        # Inputs : save_prefix = prefix of the saved image during training
        #          label = label that will display the elapsed time during training
        #          parent = parent widget

        super().__init__(parent)
        self.savePrefix = save_prefix
        self.imageLabel = QLabel(parent)
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateTimer)
        self.startTime = None
        self.elapsedLabel = label
        # flag to check if the default image is displayed
        self.flag = False
        self.parent = parent
        # removes all images except the default image
        self.removeImagesExceptDefault()
        
        # Ensure parent is a QWidget and can have a layout
        if isinstance(parent, QWidget):
            existingLayout = parent.layout
            if existingLayout is not None:
                self.layout = existingLayout
            else:
                self.layout = QVBoxLayout(parent)
                parent.setLayout(self.layout)
            
            self.layout.addWidget(self.imageLabel, alignment=Qt.AlignCenter)
        
        self.showGraphSignal.connect(self.showGraph)
        self.startTimerSignal.connect(self.startTimer)
        self.stopTimerSignal.connect(self.stopTimer)
        self.showTrainingFinishedPopUp.connect(self.popupTrainingFinishedWindow)

    # dialog pop up when training is finished
    def popupTrainingFinishedWindow(self):
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("Training Finished")
        dialog.setGeometry(100, 100, 200, 200)
        dialogLayout = QVBoxLayout(dialog)

        finishedLabel = QLabel("Training is Done")
        dialogLayout.addWidget(finishedLabel, alignment=Qt.AlignCenter)

        dialog.exec_()

    def showGraph(self):
        # Display graph in GUI
        imageFolder = './plots'
        imageFiles = [f for f in os.listdir(imageFolder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        # Find the image with the name save_prefix
        selectedImage = [f for f in imageFiles if self.savePrefix in f]
        
        # if flag is false then display default image
        if self.flag == False:
            selectedImage = "default.png"
            self.flag = True
        else:
            selectedImage = selectedImage[0]
            print('Image found')

        # If an image is found, get the first matching image
        filePath = os.path.join(imageFolder, selectedImage)
        print(f"Selected image: {filePath}")  # debug statement (remove if wanted)
        
        # Insert the selected image
        pixmap = QPixmap(filePath)
        self.imageLabel.setPixmap(pixmap)
        
        # Ensure the layout is updated
        self.layout.addWidget(self.imageLabel, alignment=Qt.AlignCenter)
        self.layout.insertWidget(0, self.imageLabel)  # Move the image label to the top of the layout

    def removeImagesExceptDefault(self):
        # assigns the directory to where all the plots are 
        imageFolder = './plots'
        for filename in os.listdir(imageFolder):
            # checks for all file names that are not "default.png"
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) and filename != "default.png":
                filePath = os.path.join(imageFolder, filename)
                os.remove(filePath)

    def startTimer(self):
        self.startTime = time.time()
        self.timer.start(1000)  # Timer interval set to 1 second
        print("Timer started.")

    def stopTimer(self):
        # Check if the timer is active
        if self.timer.isActive():
            self.timer.stop()
            elapsed_time = time.time() - self.startTime
            print(f"Timer stopped. Elapsed time: {elapsed_time:.2f} seconds.")
        else:
            # if the timer is not running and this is called print this
            print("Timer is not running.")

    def updateTimer(self):
        # update the timer every second
        elapsedTime = time.time() - self.startTime
        self.updateElapsedTime(elapsedTime)

    def updateElapsedTime(self, elapsed_time):
        # Inputs : elapsed_time = time elapsed since the timer started
        
        # convert the elapsed time to minutes and seconds
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        # change test of elapsed time label widget to the new time
        self.elapsedLabel.setText(f"Elapsed time: {minutes:02}:{seconds:02}")