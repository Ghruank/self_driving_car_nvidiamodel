from utilis import *

path="myData"
data=importDataInfo(path)

balanceData(data, display=False)

imagesPath,steering =  loadData(path,data)

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)

# print(len(xTrain), len(xVal))

model = createModel()
model.summary()

history = model.fit(batchGen(xTrain, yTrain, 10, 1), steps_per_epoch=20, epochs=2, validation_data=batchGen(xVal,yVal, 10, 0), validation_steps = 20)

model.save('model.h5')
print("model saved")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.ylim([0,1])
plt.title('loss')
plt.xlabel('epoch')
plt.show()



