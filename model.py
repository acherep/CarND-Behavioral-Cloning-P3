import csv
import cv2
import numpy as np
import sklearn

# dirrectories where driving data is stored
# driving the track forward and in the center of the lane
dir_1_forward_center = "1_data_forward_center"
# driving the track forward and on the left side of the lane
dir_1_forward_left = "1_data_forward_left"
# driving the track forward and on the right side of the lane
dir_1_forward_right = "1_data_forward_right"
epochs = 5

rows = []
with open(dir_1_forward_center + "/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # append the label to add the lane side correction in the generator
        row.append("center")
        rows.append(row)

with open(dir_1_forward_left + "/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # append the label to add the lane side correction in the generator
        row.append("left")
        rows.append(row)

with open(dir_1_forward_right + "/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
         # append the label to add the lane side correction in the generator
         row.append("right")
         rows.append(row)

# split loaded rows onto the training set (80%) and the validation set (20%)
from sklearn.model_selection import train_test_split
train_lines, validation_lines = train_test_split(rows, test_size=0.2)

# for each read row, there are 6 output images:
# one for each camera (left, right, center), 3 in total
# plus 3 corresponding flipped images
number_output_images_per_row = 6

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                lane_side = batch_sample[7]
                # based on the lane side, a new lane side correction is introduced
                lane_side_correction = 0.25
                if (lane_side == "center"):
                    # if the picture is taken in the center,
                    # there is no lane side correction
                    path = dir_1_forward_center + "/IMG/"
                    lane_side_correction = 0
                elif (lane_side == "left"):
                    # if the picture is taken to the left from the center,
                    # there is positive lane side correction
                    path = dir_1_forward_left + "/IMG/"
                elif (lane_side == "right"):
                    # if the picture is taken to the right from the center,
                    # there is negative lane side correction
                    path = dir_1_forward_right + "/IMG/"
                    lane_side_correction *= -1.0

                image_center = cv2.imread(path + batch_sample[0].split('/')[-1])
                image_left = cv2.imread(path + batch_sample[1].split('/')[-1])
                image_right = cv2.imread(path + batch_sample[2].split('/')[-1])

                # create adjusted steering angles for the side camera images
                angle_center = float(batch_sample[3])
                camera_side_correction = 0.15 # !!! should be 0.1
                # then 0.35 0.25 0.15 0.1 0 -0.1 -0.15 -0.25 -0.35
                angle_left = angle_center + camera_side_correction + lane_side_correction
                angle_right = angle_center - camera_side_correction + lane_side_correction

                # augmented images and augmented steering angles
                aug_image_center = (cv2.flip(image_center, 1))
                aug_image_left   = (cv2.flip(image_left, 1))
                aug_image_right  = (cv2.flip(image_right, 1))

                aug_angle_center = angle_center * -1.0
                aug_angle_left   = angle_left * -1.0
                aug_angle_right  = angle_right * -1.0


                # add angles and images to data sets
                images.extend([image_center, image_left, image_right, \
                               aug_image_center, aug_image_left, aug_image_right])
                angles.extend([angle_center, angle_left, angle_right, \
                               aug_angle_center, aug_angle_left, aug_angle_right])

            # reshuffle the images
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_lines, batch_size=32)
validation_generator = generator(validation_lines, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((67, 25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# from IPython.display import Image, display, SVG
# from keras.utils.visualize_util import model_to_dot

# Show the model in ipython notebook
# figure = SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
# display(figure)

# Save the model as png file
# from keras.utils.visualize_util import plot
# plot(model, to_file='model.png', show_shapes=True)

history_object = model.fit_generator( \
          train_generator, \
          samples_per_epoch=len(train_lines)*number_output_images_per_row, \
          validation_data=validation_generator, \
          nb_val_samples=len(validation_lines)*number_output_images_per_row, \
          nb_epoch=epochs)

model.save('model.h5')

print(history_object.history['loss'])
print(history_object.history['val_loss'])

### plot the training and validation loss for each epoch
import matplotlib.pyplot as plt

plt.plot(np.arange(1, epochs + 1), history_object.history['loss'])
plt.plot(np.arange(1, epochs + 1), history_object.history['val_loss'])
plt.xticks(np.arange(1, epochs + 1))
plt.title('Model mean squared error loss')
plt.ylabel('Mean squared error loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.savefig('figures/mse.png')
plt.show()