
# importing the libraries
import tensorflow as ts
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
import matplotlib.pyplot as plt
import numpy as np 

(X_trian,y_trian),(X_test,y_test) = ts.keras.datasets.cifar10.load_data()
y_trian = y_trian.flatten()
y_test = y_test.flatten()

class_names = ["Airplane","Automobile","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]

#Normalize pixel values
X_trian = X_trian/255.0
X_test = X_test/255.0

#Building/constructing the model
model = Sequential([
          Flatten(input_shape=(32,32,3)),
          Dense(256,activation="relu"),
          Dense(128,activation="relu"),
          Dense(64,activation="relu"),
          Dense(10,activation="softmax")
])

#Compiling the model
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(X_trian,y_trian,epochs=5,validation_split=0.2)

#evaluate on test data
test_loss,test_acc = model.evaluate(X_test,y_test)
print(test_acc)

num_images = 5
choice = np.random.choice(len(X_test),num_images)
predictions = model.predict(X_test[choice])

for i,idx in enumerate(choice):
    plt.figure(figsize=(2,2))
    plt.imshow(X_test[idx])
    plt.axis('off')
    predicted_label = class_names[np.argmax(predictions[i])]
    true_label = class_names[y_test[idx]]
    plt.title(f"pred:{predicted_label}\nTrue:{true_label}")
    plt.show()