import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

batch_size = 100 #size of sample in network
image_size = (250, 250)
seed = 42 # random (shuffling,transformations)

train = tf.keras.preprocessing.image_dataset_from_directory( #read every pic and store in train
    '/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',
    seed=seed,
    image_size= image_size,
    batch_size=batch_size
)
test =  tf.keras.preprocessing.image_dataset_from_directory(
    '/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid',
    seed=seed,
    image_size= image_size,
    batch_size=batch_size
)


class_names = train.class_names
print(class_names) #get class name
#cnn model
model = tf.keras.models.Sequential([
  layers.Conv2D(32, 3, activation='relu'),#
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dense(len(class_names), activation= 'softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
retVal = model.fit(train,validation_data= test,epochs = 5)

plt.plot(retVal.history['loss'], label = 'training loss')
plt.plot(retVal.history['accuracy'], label = 'training accuracy')
plt.legend()

print("Accuracy = "+str(retVal.history['accuracy'][4]))


model.save('model.h5')#save model