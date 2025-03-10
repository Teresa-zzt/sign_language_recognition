import os
import random
import sys
import threading 
import time
import cv2
import pandas as pd
import pandas
from PIL import Image
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QLineEdit, QMainWindow, QToolBar, QAction, QFileDialog,
    QListWidget, QListWidgetItem, QVBoxLayout, QHBoxLayout, QWidget, QStatusBar,
    QDesktopWidget, QSizePolicy, qApp, QGridLayout, QScrollArea, QPushButton, QSlider, QComboBox, QProgressBar,QDialog
)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from Models.Trained_model.Trained import Trainer
from Signals import Signals
from ImageScrollArea import ImageScrollArea
from PIL import ImageQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

class GUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create a window with the background colour #87AFDE (blue) along with a status bar, toolbar and search bar.
        self.createWindow() 
        self.setStyleSheet("background-color: #87AFDE;")

        self.statusBar()  
        self.createToolbar()
        self.createSearchBar()

    def createWindow(self):
        # Naming the window Sign Language Recognition System and adding in an icon
        self.setWindowTitle("Sign Language Recognition System")
        self.setWindowIcon(QIcon("src\\img\\logo.png"))
        self.move(300,300)
        self.resize(1000, 600)
        self.centre()        
        self.show()

    def createSearchBar(self):  
        # Create Search bar and layout on the top (below the toolbar)
        mainWidget = QWidget()
        # Create search bar layout
        layout = QVBoxLayout()
        mainWidget.setLayout(layout)
        self.setMenuWidget(mainWidget)

        # initalise search bar
        self.searchBar = QLineEdit()
        # set text for search bar
        self.searchBar.setPlaceholderText("Search...")
        # set style for search bar
        self.searchBar.setStyleSheet("background-color: white; border: 1px solid black; border-radius: 5px; padding: 5px;")
        # connect the search bar to the onSearch function
        self.searchBar.returnPressed.connect(self.onSearch)
        # add the widgets to our layout
        layout.addWidget(self.toolBar)
        layout.addWidget(self.searchBar)

    def createToolbar(self):
        # Create toolbar and adding in different tools
        self.toolBar = QToolBar("Main Toolbar")
        self.addToolBar(self.toolBar)

        # Exit button - exits application
        exitAction = QAction(QIcon("src\\img\\exit.png"), 'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)
        self.toolBar.addAction(exitAction)
        self.toolBar.setStyleSheet("background-color: #D9D9D9;")

        # Tool to load data into our GUI 
        loadAction = QAction("Load data", self)
        loadAction.setStatusTip("Load data")
        loadAction.triggered.connect(self.loadImages)
        self.toolBar.addAction(loadAction)
        
        # Where the loaded data is viewed
        viewAction = QAction("View", self)
        viewAction.setStatusTip("View")
        viewAction.triggered.connect(self.viewActionOnClickListener)
        self.toolBar.addAction(viewAction)

        # Can adjust which model and how it should be trained
        trainAction = QAction("Train", self)
        trainAction.setStatusTip("Train Dataset")
        trainAction.triggered.connect(self.trainActionOnClickListener)
        self.toolBar.addAction(trainAction)

        # Tests the trained model and output the results
        testAction = QAction("Test", self)
        testAction.setStatusTip("Test Dataset")
        testAction.triggered.connect(self.testActionOnClickListener)
        self.toolBar.addAction(testAction)

        # Opens webcam
        camAction = QAction("Camera", self)
        camAction.setStatusTip("Camera")
        camAction.triggered.connect(self.showCam)
        self.toolBar.addAction(camAction)

    def viewActionOnClickListener(self):
        # lists the images loaded
        folderPath = "CSV_Images"
        imageFiles = [f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f)) and f.endswith('.png')]
        self.scrollArea = ImageScrollArea(folderPath, imageFiles)
        self.setCentralWidget(self.scrollArea)
        self.addSearchBar()

    def trainActionOnClickListener(self):
        # Shows the training interface which doesnt have a searchbar
        self.showTrainingInterface()
        self.removeSearchBar()

    def testActionOnClickListener(self):
        # Shows the testing interface
        testingWidget = QWidget()
        self.layout = QVBoxLayout(testingWidget)

        label = QLabel("Select a Model")
        label.setStyleSheet("font-weight: bold;")
        self.layout.addWidget(label, alignment=Qt.AlignCenter)
        
        # Add a dropdown list to select the model
        self.dropdown = QComboBox()
        modelFolderPath = "checkpoints"
        modelFiles = [f for f in os.listdir(modelFolderPath) if os.path.isfile(os.path.join(modelFolderPath, f)) and f.endswith('.pt')]
        self.dropdown.addItems(modelFiles)
        self.dropdown.setStyleSheet("background-color: #FFFFFF; color: black;")
        # generate the new model path and model type for the testing when the dropdown is changed
        self.dropdown.currentIndexChanged.connect(self.updateTestModel)
        self.layout.addWidget(self.dropdown)
        self.updateTestModel()

        self.layout.addStretch()
        # add select image button for personally taken photos via. webcam
        self.selectImage = QPushButton('Select Image')
        self.selectImage.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.selectImage.setFixedSize(200, 50)
        self.selectImage.clicked.connect(self.openImage)
        self.layout.addWidget(self.selectImage, alignment=Qt.AlignCenter)
        

        # Create a scroll area to display images
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaContent = QWidget()
        self.scrollAreaLayout = QGridLayout(self.scrollAreaContent)  # Use QGridLayout
        self.scrollArea.setWidget(self.scrollAreaContent)
        self.layout.addWidget(self.scrollArea)
        # Set the folder containing the images
        self.imageFolder="CSV_Images"
        # create a list to store the displayed images, to avoid duplicates
        self.displayedImages = []
        # Load the initial set of images
        self.loadMoreImages()

        # Connect the scroll area to check if the scroll bar reach the bottom
        self.scrollArea.verticalScrollBar().valueChanged.connect(self.checkScrollPosition)

        self.layout.addStretch()
        
        self.setCentralWidget(testingWidget)

        self.removeSearchBar()

    def openImage(self):
        # Open a file dialog to select an image
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        imagePath, _ = QFileDialog.getOpenFileName(self, "Select Image", "./test_capture_image", "Image Files (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if imagePath:
            self.showImagePopup(imagePath)

    def loadMoreImages(self):
        # Load 100 more images when the scroll bar reaches the bottom
        # find all the images in the imageFolder
        imageFiles = [f for f in os.listdir(self.imageFolder) if f.lower().endswith(('.png'))]
        # randomly select 100 images
        newImages = random.sample(imageFiles, 100)
        row = self.scrollAreaLayout.rowCount()  # Get the current row count
        col = 0
        # Add the images to the scroll area
        for i,imageFile in enumerate(newImages):
            if imageFile not in self.displayedImages:
                # Add the image to the list of displayed images to avoid duplicates
                self.displayedImages.append(imageFile)
                # Get the path of the current image
                imagePath = os.path.join(self.imageFolder, imageFile)
                # Create a label to display the image
                imageLabel = QLabel()
                pixmap = QPixmap(imagePath)
                # Display the image in the label with size 100x100
                imageLabel.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                imageLabel.mousePressEvent = lambda event, img_path=imagePath: self.showImagePopup(img_path)
                row = (self.scrollAreaLayout.count() + i) // 10
                col = (self.scrollAreaLayout.count() + i) % 10
                self.scrollAreaLayout.addWidget(imageLabel, row, col)

    def checkScrollPosition(self):
        # Check if the scroll bar has reached the button, if reached, load 100 more images
        scrollBar = self.scrollArea.verticalScrollBar()
        if scrollBar.value() == scrollBar.maximum():
            self.loadMoreImages()

    def updateTestModel(self):
        # Update the model path and model type when the dropdown is changed
        self.modelPath = self.dropdown.currentText()
        self.modelType = self.modelPath.split('_')[0]

    def showImagePopup(self, imagePath):
        # Inputes: imagePath = the path of the image to display

        # Display the image and confidence levels in a popup
        dialog = QDialog(self)
        dialog.setWindowTitle("Image Details")
        dialog.setGeometry(300, 300, 400, 400)
        dialogLayout = QVBoxLayout(dialog)

        # Display the image
        imageLabel = QLabel()
        pixmap = QPixmap(imagePath)
        imageLabel.setPixmap(pixmap.scaled(600, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        dialogLayout.addWidget(imageLabel)

        # create a current train class and called the inference method to get the top 5 possible classes
        confidenceLabel = QLabel()
        train=Trainer(self.modelType, "./data_images")
        # convert the image to RGB for prediction
        image=Image.open(imagePath).convert('RGB')
        confidenceList=train.inference(self.modelPath,image)
        # confidence_list is the top 5 possible classes for the image
        confidenceText = "\n".join(confidenceList)
        confidenceLabel.setText(confidenceText)
        dialogLayout.addWidget(confidenceLabel)

        # show pop up dialog
        dialog.exec_()

    def showCam(self):  
        # Open a connection to the default camera (usually the first camera)
        cap = cv2.VideoCapture(0)
        
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        capture_count = 0
        capture_folder = "test_capture_image"

        # Ensure the capture folder exists
        if not os.path.exists(capture_folder):
            os.makedirs(capture_folder)

        # Read and display frames in a loop
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Display the frame in a window named 'Webcam'
            cv2.imshow('Webcam', frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            # Exit the window when 'q' key is pressed or the window is closed
            if key == ord('q') or cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1:
                break
            
            # Capture and save the frame when 'c' key is pressed
            if key == ord('c'):
                capture_count += 1
                filename = os.path.join(capture_folder, f"capture_{capture_count}.png")
                cv2.imwrite(filename, frame)
                self.capturedImagePopUp()
                print(f"Captured image saved as {filename}")
        
        # Release the webcam and close the window
        cap.release()
        cv2.destroyAllWindows()

    def removeSearchBar(self):
        # Removes search bar
        self.searchBar.hide()
    
    def addSearchBar(self):
        # Adds in search bar
        self.searchBar.show()
    

    def loadImages(self):
        # Prompts the user to choose the file they want and load the content
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        # Open a file dialog to select a CSV file
        filePath, _ = QFileDialog.getOpenFileName(self, "Select File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        # Check if a file was selected
        if filePath:
            # Check if the file is a CSV file
            if filePath.endswith('.csv'):
                df = pd.read_csv(filePath)
                print(f"Started conversion of CSV file: {filePath}")
                self.thread2 = threading.Thread(self.convertCSVToImages(df))
                self.thread2.start()
            # Create a folder to save the training and test images
                rootDir = 'data_images'
                if os.path.exists(rootDir):
                    import shutil

                    shutil.rmtree(rootDir)
                    print('Removed existing ImageFolder structure.')

                if not os.path.exists(rootDir):
                    os.makedirs(rootDir)

                data = pd.read_csv(filePath)
                X = data.iloc[:, 1:].values.astype(np.uint8)
                y = data.iloc[:, 0].values.astype(np.int64)
                print('Class distribution:')
                print(pandas.Series(y).value_counts())

                for index, (image, label) in enumerate(zip(X, y)):
                    labelDir = os.path.join(rootDir, f"{label:02}")
                    if not os.path.exists(labelDir):
                        os.makedirs(labelDir)

                    image = image.astype(np.uint8).reshape(28, 28)
                    image = Image.fromarray(image)
                    image.save(os.path.join(labelDir, f'{index + 1}.png'))
                return filePath  # Return the file path
        return None

    def convertCSVToImages(self, df):
        # Inputs: df = the DataFrame containing the image data

        # Create a folder to save the images
        folderName = "CSV_Images"
        os.makedirs(folderName, exist_ok=True)
        print("Folder created:", folderName)

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            # Exclude the label (assuming it's the first column)
            imageData = row.values[1:].astype(np.uint8)
            label = row.values[0]

            # Define the image size (assuming it's a square image)
            size = int(np.sqrt(len(imageData)))

            # Reshape the data into an image
            try:
                imageData = imageData.reshape((size, size))
            except ValueError as e:
                print(f"Error reshaping row {index}: {e}")
                continue

            # Create an image from the array
            image = Image.fromarray(imageData)

            # Save the image to the folder
            imageFileName = os.path.join(folderName, f"{label}_{index}.png")
            image.save(imageFileName)
        self.csvToImagesConversionPopUp()
        print("Images created and saved successfully.")


    # Takes in user input and displays search results
    def onSearch(self):
        searchText = self.searchBar.text().strip()
        if searchText.isdigit():
            searchNumber = searchText
            folderPath = "CSV_Images"
            imageFiles = [f for f in os.listdir(folderPath)
                           if os.path.isfile(os.path.join(folderPath, f)) and f.endswith('.png') and f.startswith(f"{searchNumber}_")]
            self.scrollArea = ImageScrollArea(folderPath, imageFiles)
            self.setCentralWidget(self.scrollArea)

    def centre(self):
        # Centers the window on user monitor/display
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def showTrainingInterface(self):
        # Shows training interface and add/remove features on interface
        trainingWidget = QWidget()
        
        # Check if the widget already has a layout, and if so, remove it
        if trainingWidget.layout() is not None:
            while trainingWidget.layout().count():
                item = trainingWidget.layout().takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            trainingWidget.layout().deleteLater()

        self.layout = QVBoxLayout(trainingWidget)

        # Add a label above the dropdown
        label = QLabel("Select a model")
        label.setStyleSheet("font-weight: bold;")  # Set font weight to bold
        self.layout.addWidget(label, alignment=Qt.AlignCenter)

        # Add the dropdown
        self.dropdown = QComboBox()
        # Add items to the dropdown 
        self.dropdown.addItems(["Select Model", "Custom", "MnasNet", "MobileNet"])  
        self.layout.addWidget(self.dropdown)
        self.dropdown.setStyleSheet("background-color: #FFFFFF; color: black;")

        # Add a stretch before the button to move it higher
        self.layout.addStretch()

        # time left label
        # initalise the elapsed time label
        self.elapsedTime = QLabel("Elapsed Time: 00:00")

        # initalise progress bar
        # self.progressBar = QProgressBar()
        # self.progressBar.setRange(0, 100)
        
        # Add the "Start Training" button
        self.startTrainingButton = QPushButton('Start Training')
        self.startTrainingButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.startTrainingButton.setFixedSize(200, 50)  # Set fixed size for the button

        # Connect the training thread to the button
        self.startTrainingButton.clicked.connect(self.startTrainingThread) 
        
        self.layout.addWidget(self.startTrainingButton, alignment=Qt.AlignCenter)  # Center the button
        self.startTrainingButton.setStyleSheet("background-color: #FFFFFF; color: black;")  # Set button color
        
        # Add a stretch before the button to move it higher
        self.layout.addStretch()

        # Add a label above the "Train/Test Ratio" slider
        label = QLabel("Train/Test Ratio")
        label.setStyleSheet("font-weight: bold;")  # Set font weight to bold
        self.layout.addWidget(label, alignment=Qt.AlignCenter)
        
        # Add the "Train/Test Ratio" slider
        self.ratioSlider = QSlider(Qt.Horizontal)
        self.ratioSlider.setMinimum(5)
        self.ratioSlider.setMaximum(95)
        self.ratioSlider.setTickInterval(1)  # Set the tick interval to 0.1
        self.ratioSlider.setSingleStep(5)    # Set the single step to 0.1
        self.ratioSlider.setTickPosition(QSlider.TicksBelow)  # Optional: Show ticks below the slider
        self.ratioSlider.setStyleSheet("background-color: #FFFFFF; color: black;")
        self.ratioSlider.valueChanged.connect(self.updateLabelRatio)
        self.layout.addWidget(self.ratioSlider)
        self.ratioLabel = QLabel()
        self.layout.addWidget(self.ratioLabel)
        self.updateLabelRatio()

        # Add a stretch after the slider
        self.layout.addStretch()

        # Add a label above the "Batch Size" slider
        label = QLabel("Batch Size")
        label.setStyleSheet("font-weight: bold;")  # Set font weight to bold
        self.layout.addWidget(label, alignment=Qt.AlignCenter)
        
        # Add the "Batch Size" slider
        self.batchSlider = QSlider(Qt.Horizontal)
        self.batchSlider.setMinimum(0)
        self.batchSlider.setMaximum(4)  # Scale from 0 to 4
        self.batchSlider.setTickInterval(1)
        self.batchSlider.setSingleStep(1)
        self.batchSlider.setTickPosition(QSlider.TicksBelow)
        self.batchSlider.setStyleSheet("background-color: #FFFFFF; color: black;")
        self.batchSlider.valueChanged.connect(self.updateLabelBatch)
        # sent the value of the batch_size to the label
        self.layout.addWidget(self.batchSlider)
        self.batchLabel = QLabel()
        self.layout.addWidget(self.batchLabel)
        self.updateLabelBatch()
        # Add a stretch after the slider
        self.layout.addStretch()

        # Add a label above the "Epochs" slider
        label = QLabel("Epochs")
        label.setStyleSheet("font-weight: bold;")  # Set font weight to bold
        self.layout.addWidget(label, alignment=Qt.AlignCenter)
        
        # Add the "Epochs" slider
        self.epochsSlider = QSlider(Qt.Horizontal)
        self.epochsSlider.setMinimum(1)
        self.epochsSlider.setMaximum(50)
        self.epochsSlider.setTickInterval(1)  # Set the tick interval to 1
        self.epochsSlider.setSingleStep(1)    # Set the single step to 1
        self.epochsSlider.setTickPosition(QSlider.TicksBelow)  # Optional: Show ticks below the slider
        self.epochsSlider.setStyleSheet("background-color: #FFFFFF; color: black;")
        self.epochsSlider.valueChanged.connect(self.updateLabelEpoch)
        self.layout.addWidget(self.epochsSlider)
        self.epochLabel = QLabel()
        self.layout.addWidget(self.epochLabel)
        self.updateLabelEpoch()

        # Add a stretch after the slider
        self.layout.addStretch()
        self.setCentralWidget(trainingWidget)
    
    # update the ratio label as slider moves
    def updateLabelRatio(self):
        value = self.ratioSlider.value()
        self.ratioLabel.setText(f'The testing dataset ratio: {value}%')

    # update batch label as the slider moves
    def updateLabelBatch(self):
        batchSizes = [16, 32, 64, 128, 256]
        value = batchSizes[self.batchSlider.value()]
        self.batchLabel.setText(f'Batch size: {value}')

    # update the epoch label as the slider moves
    def updateLabelEpoch(self):
        value = self.epochsSlider.value()
        self.epochLabel.setText(f'Number of Epoch: {value}')

    # initalise the batch sizes
    def batchSize(self):
        batchSizes = [16, 32, 64, 128, 256]
        return batchSizes[self.batchSlider.value()]
    
    def updateInterfaceForTraining(self):
        # Remove all widgets from the layout
        for i in reversed(range(self.layout.count())): 
            widgetToRemove = self.layout.itemAt(i).widget()
            if widgetToRemove is not None: 
                self.layout.removeWidget(widgetToRemove)
                widgetToRemove.setParent(None)
        
        # Show graph
        self.signals.showGraphSignal.emit()

        # Create a horizontal layout for the stop button and progress bar
        hLayout = QHBoxLayout()

        # Add the "Stop Training" button
        self.stopTrainingButton = QPushButton('Stop Training')
        self.stopTrainingButton.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stopTrainingButton.setFixedSize(200, 50)  # Set fixed size for the button
        hLayout.addWidget(self.stopTrainingButton, alignment=Qt.AlignCenter)
        self.stopTrainingButton.clicked.connect(self.showTrainingInterface)  # Connect the button to the VGG function
        self.stopTrainingButton.clicked.connect(self.stopTraining)
        self.stopTrainingButton.setStyleSheet("background-color: #FFFFFF; color: black;")  # Set button color
        
        # hLayout.addWidget(self.progressBar)

        self.layout.addWidget(self.elapsedTime, alignment=Qt.AlignCenter)

        # Add the horizontal layout to the main layout
        self.layout.addLayout(hLayout)

        # Update the elapsed time label to indicate training is in progress
        self.layout.addWidget(self.elapsedTime, alignment=Qt.AlignCenter)

    # thread to allow responsive window while training
    def startTrainingThread(self):
        batchSizeValue = self.batchSize()
        # check what model the user selected
        if (self.dropdown.currentText() == "Custom" or self.dropdown.currentText() == "MnasNet" or self.dropdown.currentText() == "MobileNet"):
            # save the name of the model and plot
            self.savePrefix = f'{self.dropdown.currentText()}_bs{batchSizeValue}_epochs{self.epochsSlider.value()}'
            # initalise all signals to be able to change widgets during threading
            self.signals = Signals(self.savePrefix, self.elapsedTime, parent=self)

            # change the layout to display training information
            self.updateInterfaceForTraining() 
            # make instance of trainer class
            self.train = Trainer(self.dropdown.currentText(), "./data_images", self.signals, batchSizeValue, 0.0001, self.epochsSlider.value(), (self.ratioSlider.value()/100))
            self.thread1 =  threading.Thread(target=self.trainModel)# Connect the button to the Inception function
            self.thread1.start()
            # start timer
            self.signals.startTimerSignal.emit()
        else:
            print("No model selected")

    def stopTraining(self):
        # stop training
        self.train.request_stop()
        self.signals.stopTimerSignal.emit()

    def trainModel(self):
        # Trains model
        self.train.train()
        self.signals.stopTimerSignal.emit()
        self.signals.showTrainingFinishedPopUp.emit()
        print("end of train model function")

    # dialog pop up for when our csv file (dataset) is finished converting to images
    def csvToImagesConversionPopUp(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Image Details")
        dialog.setGeometry(900, 400, 200, 200)
        dialogLayout = QVBoxLayout(dialog)

        finishedLabel = QLabel("Conversion Done")
        dialogLayout.addWidget(finishedLabel, alignment=Qt.AlignCenter)

        dialog.exec_()

    # dialog pop up for  when a image is captured
    def capturedImagePopUp(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Captured Image")
        dialog.setGeometry(900, 400, 200, 200)
        dialogLayout = QVBoxLayout(dialog)

        finishedLabel = QLabel("Image Captured")
        dialogLayout.addWidget(finishedLabel, alignment=Qt.AlignCenter)

        dialog.exec_()

    
     
# allows this file to run 
if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = GUI()
   sys.exit(app.exec_())


