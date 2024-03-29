import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.callbacks import ModelCheckpoint
# Load the fashion-mnist pre-shuffled train data and test data
X_tr = np.load("fashion_mnist_train_images.npy")
x_train = X_tr.reshape(X_tr.shape[0],28,28)
y_train = (np.load("fashion_mnist_train_labels.npy"))

X_te = np.load("fashion_mnist_test_images.npy")
x_test = X_te.reshape(X_te.shape[0],28,28)
y_test = (np.load("fashion_mnist_test_labels.npy"))


#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Print training set shape - note there are 60,000 training data of image size of 28x28, 60,000 train labels)
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training and test datasets
print(x_train.shape[0], 'train set')
print(x_test.shape[0], 'test set')

# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9



x_train = x_train.astype('float32') / 255

x_test = x_test.astype('float32') / 255


print("Number of train data - " + str(len(x_train)))
print("Number of test data - " + str(len(x_test)))


# Further break training data into train / validation sets (# put 5000 into validation set and keep remaining 55,000 for train)
(x_train, x_valid) = x_train[5000:], x_train[:5000] 
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')

#model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network

x = tf.keras.layers.Input(shape=(28,28,1))

conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu')(x)
btc1 = tf.keras.layers.BatchNormalization()(conv1)
conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(btc1)
btc2 = tf.keras.layers.BatchNormalization()(conv2)
conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu')(btc2)
btc3 = tf.keras.layers.BatchNormalization()(conv3)
drp1 = tf.keras.layers.Dropout(0.4)(btc3)

conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(drp1)
btc4 = tf.keras.layers.BatchNormalization()(conv4)
conv5 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(btc4)
btc5 = tf.keras.layers.BatchNormalization()(conv5)
conv6 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same', activation='relu')(btc5)
btc6 = tf.keras.layers.BatchNormalization()(conv6)
drp2 = tf.keras.layers.Dropout(0.4)(btc6)

conv7 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(drp2)
btc7 = tf.keras.layers.BatchNormalization()(conv7)
conv8 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(btc7)
btc8 = tf.keras.layers.BatchNormalization()(conv8)
conv9 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu')(btc8)
btc9 = tf.keras.layers.BatchNormalization()(conv9)

drp3 = tf.keras.layers.Dropout(0.4)(btc9)
conv10 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu')(drp3)
btc10 = tf.keras.layers.BatchNormalization()(conv10)
conv11 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu')(btc10)
btc11 = tf.keras.layers.BatchNormalization()(conv11)



#model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu', input_shape=(28,28,1))) 
#model.add(tf.keras.layers.MaxPooling2D(pool_size=2,padding='same'))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu') )
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides = 2, padding='same', activation='relu')) 
#model.add(tf.keras.layers.BatchNormalization())

#model.add(tf.keras.layers.Dropout(0.4))

#
#model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu') )
#model.add(tf.keras.layers.MaxPooling2D(pool_size=2,padding='same'))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu') )
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same', activation='relu')) 
#model.add(tf.keras.layers.BatchNormalization())

#model.add(tf.keras.layers.Dropout(0.4))


#model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu') )
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu') )
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=5, padding='same', activation='relu') )
#model.add(tf.keras.layers.BatchNormalization())
	
#model.add(tf.keras.layers.Dropout(0.4))


#model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=5, padding='same', activation='relu') )
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=5, padding='same', activation='relu') )
#model.add(tf.keras.layers.BatchNormalization())


merge = tf.keras.layers.Add()([btc9,btc11])

drp4 = tf.keras.layers.Dropout(0.4)(merge)

conv12 = tf.keras.layers.Conv2D(filters=256, kernel_size=7, strides=2, padding='same', activation='relu')(drp4)
btc12 = tf.keras.layers.BatchNormalization()(conv12)

flat = tf.keras.layers.Flatten()(btc12)
relu1 = tf.keras.layers.Dense(512, activation='relu')(flat)
btc13 = tf.keras.layers.BatchNormalization()(relu1)
drp4 = tf.keras.layers.Dropout(0.5)(btc13)
sfm1 = tf.keras.layers.Dense(10, activation='softmax')(drp4)

model = tf.keras.Model(inputs = x, outputs = sfm1)

#model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=7, padding='same', activation='relu') )
#model.add(tf.keras.layers.BatchNormalization())


#model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.Dense(512, activation='relu'))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dropout(0.5))
#model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Take a look at the model summary
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(loss='categorical_crossentropy',
             optimizer=opt,
             metrics=['accuracy'])



#checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)
model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=10,
         validation_data=(x_valid, y_valid))

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])

y_hat = model.predict(x_test)


# Plot a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index], 
                                  fashion_mnist_labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))

#plt.show()
print("training_ended")
#print(model.summary())


