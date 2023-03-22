import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
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

# Image index, you can pick any number between 0 and 59,999
img_index = 5
# y_train contains the lables, ranging from 0 to 9
label_index = y_train[img_index]
# Print the label, for example 2 Pullover
print ("y = " + str(label_index) + " " +(fashion_mnist_labels[label_index]))
# # Show one of the images from the training dataset
plt.imshow(x_train[img_index])


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

model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', activation=None, input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='valid'))
model.add(tf.keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0))

#model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Take a look at the model summary
model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)
model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=1,
         validation_data=(x_valid, y_valid),
         callbacks=[checkpointer])

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])

testx = x_test[1]
testx = testx[np.newaxis]
y_hat = model.predict(testx)

print(y_hat)

y_hat = model.predict(x_test)

model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', activation=None, input_shape=(28,28,1)))
model2.add(tf.keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='valid'))
model2.add(tf.keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
model2.add(tf.keras.layers.Flatten())


#set weights of the first layer
model2.set_weights(model.layers[0].get_weights())

#compile it after setting the weights
model2.compile(optimizer='adam', loss='categorical_crossentropy')

hidden_state = model2.predict(testx)
hidden_state = np.moveaxis(hidden_state,-1,0)




#print(hidden_state[63])


W1 = np.asarray(model.trainable_variables[0])
b1 = np.asarray(model.trainable_variables[1])
W2 = np.asarray(model.trainable_variables[2])
b2 = np.asarray(model.trainable_variables[3])
W3 = np.asarray(model.trainable_variables[4])
b3 = np.asarray(model.trainable_variables[5])

W1 = np.moveaxis(W1,-1,0)

w1_layer = np.zeros((43264,784))



for j in range(43264):
	for k in range(3):
		for l in range(3):
				ind = k*28+l+(math.floor(((j%676)/26))*28) + j%26
				w1_layer[j,ind] = W1[math.floor(j/676),k,l]


b1_layer = np.zeros((43264))

for i in range(43264):
	b1_layer[i] = b1[math.floor(i/676)]

weights_conv = [w1_layer,W2,W3]

bias_conv = [b1_layer,b2,b3]

weights_bias_conv = [weights_conv,bias_conv] # np.hstack([ weights.flatten() for weights in weights_conv] + [ bias.flatten() for bias in bias_conv])


with open("test", "wb") as fp:   #Pickling
	pickle.dump(weights_bias_conv, fp)


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

print("training_ended")

