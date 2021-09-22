import cv2 as cv
import numpy as np

model = cv.dnn.readNetFromONNX('CornholeBBBestV1.onnx')

filePath = os.path.join("Images", "Game4.jpg")
image = cv.imread(filePath)
blob = cv.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224), mean=(104, 117, 123))

model.setInput(blob)
outputs = model.forward()

final_outputs = outputs[0]
final_outputs = final_outputs.reshape(1000, 1)
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
final_prob = np.max(probs) * 100.
cv.putText(image, "hehe", (25, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv.imshow('Image', image)
cv.waitKey(0)
cv.imwrite('result_image.jpg', image)
