import cv2
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt

bridge016 = "C:\\Users\\Hannes\\projects\\VC\\pictures\\bridge\\bridge_500.jpg"
bridge2 = "C:\\Users\\Hannes\\projects\\VC\\pictures\\bridge\\bridge_501.jpg"
bridge3 = "C:\\Users\\Hannes\\projects\\VC\\pictures\\bridge\\bridge_502.jpg"
bridge4 = "C:\\Users\\Hannes\\projects\\VC\\pictures\\bridge\\bridge_503.jpg"


def rgbPlot(img):
    redHist = (cv2.calcHist([img], [0], None, [256], [0, 256]))
    greenHist = (cv2.calcHist([img], [1], None, [256], [0, 256]))
    blueHist = (cv2.calcHist([img], [2], None, [256], [0, 256]))
    max = np.max([np.max(redHist), np.max(greenHist), np.max(blueHist)])
    redScaledCumHist = scaleYaxis(cumulativeHist(redHist), max / redHist.sum())
    greenScaledCumHist = scaleYaxis(cumulativeHist(greenHist), max / greenHist.sum())
    blueScaledCumHist = scaleYaxis(cumulativeHist(blueHist), max / blueHist.sum())
    plt.plot(redHist, color="r")
    plt.plot(greenHist, color="g")
    plt.plot(blueHist, color="b")
    plt.plot(redScaledCumHist, color="r", linestyle="dashed", alpha=0.5)
    plt.plot(greenScaledCumHist, color="g", linestyle="dashed", alpha=0.5)
    plt.plot(blueScaledCumHist, color="b", linestyle="dashed", alpha=0.5)


def scaleYaxis(array, factor):
    newArray = np.array(array)
    for i in range(0, len(newArray)):
        newArray[i] = factor * array[i]
    return newArray


def cumulativeHist(hist):
    cumHist = np.array(hist)
    for i in range(1, len(hist)):
        cumHist[i] = cumHist[i - 1] + hist[i]
    return cumHist


def imageLayGrid(originalImage, tileNumber):
    img = np.array(originalImage)
    sizeY = len(img)
    sizeX = len(img[0])
    tilesizeY = int(sizeY / tileNumber)
    tilesizeX = int(sizeX / tileNumber)
    tileArray = np.empty((tileNumber, tileNumber), type(img))
    for row in range(0, tileNumber):
        cv2.line(img, (0, tilesizeX * row), (sizeY, tilesizeX * row), (255, 0, 0), 1, 1)
        cv2.line(img, (tilesizeY * row, 0), (tilesizeY * row, sizeX), (255, 0, 0), 1, 1)
        for col in range(0, tileNumber):
            tileArray[row, col] = originalImage[tilesizeY * row:tilesizeY * row + tilesizeY,
                                  tilesizeX * col:tilesizeX * col + tilesizeX]
    return img, tileArray


img = cv2.imread(bridge2)
rgbPlot(img)
plt.show()
gridded, tileArray = imageLayGrid(img, 3)
plt.imshow(img)
plt.show()
plt.imshow(tileArray[1, 2])
plt.show()
