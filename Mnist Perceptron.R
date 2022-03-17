library(data.table)
library(mltools)

# Path to folder containing all data.
DATA_PATH = "Data"

# Training data
TRAIN_LABELS_FILE <- paste(DATA_PATH, "/", "train-labels.idx1-ubyte", sep="")
TRAIN_IMAGES_FILE <- paste(DATA_PATH, "/", "train-images.idx3-ubyte", sep="")

# Test data
TEST_LABELS_FILE <- paste(DATA_PATH, "/", "t10k-labels.idx1-ubyte", sep="")
TEST_IMAGES_FILE <- paste(DATA_PATH, "/", "t10k-images.idx3-ubyte", sep="")


# Constants
HEIGHT = 0
WIDTH = 0
NUM_TRAIN_SIZE = 0
NUM_TEST_SIZE = 0

to.read = file(TRAIN_IMAGES_FILE, "rb")

magic_number = readBin(to.read, integer(), n=1, endian="big")
num_train_labels = readBin(to.read, integer(), n=1, endian="big")
height = readBin(to.read, integer(), n=1, endian="big")
width = readBin(to.read, integer(), n=1, endian="big")


HEIGHT = height
WIDTH = width
IMAGE_PIXELS = HEIGHT * WIDTH
NUM_TRAIN_SIZE = num_train_labels


par(mfrow=c(5,5))
par(mar=c(0,0,0,0))
for(i in 1:24){m = matrix(readBin(to.read,integer(), size=1, n=IMAGE_PIXELS, endian="big"),HEIGHT, WIDTH);image(m[,HEIGHT:1])}
image(m)

# RELU activation
RELU <- function(x) {
  result <- ifelse(x < 0, 0, x)
  return (result)
}


RELU(c(1,2, 0 , -3))


# One hot encoding
one_hot_encode <- function(labels) {
  data_frame = data.frame(label=labels)
  data_frame$label <- as.factor(data_frame$label)
  return (one_hot(as.data.table(data_frame)) )
}

one_hot_encode(c(1, 2, 3, 5, 7))



# Function for reading labels
read_labels <- function(filename) {
  to.read.labels = file(filename, "rb")
  # Magic number
  readBin(to.read.labels, integer(), n=1, endian="big")
  # Number of labels
  num_labels <- readBin(to.read.labels, integer(), n=1, endian="big")
  
  # Following values are labels
  labels <- c()
  for (i in 1:num_labels) {
    label = as.integer(readBin(to.read.labels, raw(), n=1, endian="big"))
    labels <- append(labels, label)
  }
  
  return (labels)
}


# Function for reading image data.
read_images <- function(filename) {
  to.read.images = file(filename, "rb")
  # Magic number
  readBin(to.read.images, integer(), n=1, endian="big")
  
  # Number of images
  num_images = readBin(to.read.images, integer(), n=1, endian="big")
  
  # Number of rows
  num_rows = readBin(to.read.images, integer(), n=1, endian="big")
  
  # Number of columns
  num_columns = readBin(to.read.images, integer(), n=1, endian="big")
  
  size_images_block = num_rows * num_columns * num_images
  
  images <- list()
  
  raw <- readBin(to.read.images, raw(), n=size_images_block, endian="big")
  close(to.read.images)
  sel <- rep(0:(num_images - 1), each=IMAGE_PIXELS) * IMAGE_PIXELS + 1:IMAGE_PIXELS
  raw_images <- readBin(raw[sel], what="raw", n=size_images_block, endian="big")
  raw_images <- as.integer(raw_images)
  
  # If you want to keep the raw values, remove the abs-call.
  raw_images <- abs(raw_images / 255)
  return (raw_images)
}

# Show some example data.
labels_train <- read_labels(TRAIN_LABELS_FILE)
labels_train <- one_hot_encode(labels_train)
labels_train[1:10]

train_images <- read_images(TRAIN_IMAGES_FILE)
print(paste("Length of train images: ", length(train_images) / IMAGE_PIXELS))

m = matrix( train_images[1:IMAGE_PIXELS], HEIGHT, WIDTH)
image(m)


rotate_image <- function(image_raw) {
  m = matrix(image_raw, HEIGHT, WIDTH)
  m = m[,HEIGHT:1]
  return (m)
}


test_labels <- read_labels(TEST_LABELS_FILE)
test_labels <- one_hot_encode(test_labels)
test_labels[1:10]

test_images <- read_images(TEST_IMAGES_FILE)
print(paste("Length of test images:", length(test_images) / IMAGE_PIXELS))



m = matrix(test_images[1:IMAGE_PIXELS],HEIGHT, WIDTH)
m = m[,HEIGHT:1]
image(m)


NUM_OUTPUTS = 10
NUM_INPUTS = IMAGE_PIXELS + 1 # Note we are using a bias input.

# Function for generating random weights matrix.
generate_weight_matrix <- function() {
  matrix(runif( (NUM_INPUTS * NUM_OUTPUTS), -0.1, 0.1), nrow=NUM_INPUTS, ncol=NUM_OUTPUTS)
}



WEIGHTS = generate_weight_matrix()
WEIGHTS
print(paste("Dimension of randomly created weights matrix: ", dim(WEIGHTS)))


# Function for prediciton.
predict <- function(image, W) {
  #' This function returns a prediction, about a given input image.
  
  # Add bias input
  input_with_bias <- append(image, 1)
  
  # Calculate activation
  act <- input_with_bias %*% W
  
  # Calculate activation function
  out = RELU(act)
  
  # Return out
  return (out)
}


# Function for training.
train <- function(input, output, t, W) {
  # Calculate error.
  error <- t - output
  learning_rate <- 0.001
  # Update weights-matrix.
  for (i in 1:NUM_OUTPUTS) {
    for (j in 1:NUM_INPUTS) {
      delta <- learning_rate * error[i] * input[j]
      if (delta != 0) {
        W[j, i] <- W[j, i] + delta
      }
    }
  }
  
  return (W)
}


# Function for testing.
test <- function(inputs, teacher_outputs, W) {
  
  # Get information about the input.
  nr_samples <- length(inputs) / IMAGE_PIXELS
  correct <- 0
  
  start_index <- 1
  end_index <- IMAGE_PIXELS
  
  for (i in 1:nr_samples) {
    
    # Get respective data.
    image <- as.vector ( rotate_image( inputs[start_index:end_index] ) )
    label <- teacher_outputs[i]
    
    # Calculate output.
    out <- predict(image, W)
    
    # Get prediction and ground truth.
    prediction <- which.max(out) - 1
    gt_label <- which.max( unname(unlist(label)) ) - 1
    
    # Update correct guessed if prediction matches ground truth.
    if (prediction == gt_label) {
      correct <- correct + 1
    }
    
    # Update start and end index.
    start_index <- end_index + 1
    end_index <- start_index + IMAGE_PIXELS - 1
  }
  
  # Calculate and return accuracy.
  accuracy <- correct / nr_samples
  return (accuracy)
}




# Main-Loop

main_loop <- function() {
  accuracies <- c()
  epochs <- 10000
  weights <- generate_weight_matrix()
  nr_train_samples <- length(train_images) / IMAGE_PIXELS
  for (epoch in 1:epochs) {
    
    random_index <- as.integer(runif(1, 1, nr_train_samples))
    start_index <- IMAGE_PIXELS * (random_index - 1) + 1
    end_index <- start_index + IMAGE_PIXELS - 1
    
    image <- rotate_image ( train_images[start_index:end_index] )
    m <- matrix(image, HEIGHT, WIDTH)
    image <- as.vector ( rotate_image ( train_images[start_index:end_index] ) )
    label <- unname( unlist( labels_train[random_index] ) )
    
    prediction <- predict(image, weights)
    
    predicted_label <- which.max(prediction) - 1
    gt_label <- which.max(label) - 1
    
    
    image_with_bias <- append(image, 1)
    
    weights <- train(image_with_bias, prediction, label, weights)
    
    # Show current accuracy on test set every 100 epochs.
    if (epoch %% 100 == 0) {
      acc <- test(test_images, test_labels, weights)
      print(paste("Epoch ", epoch, " accuracy on test: ", acc))
      accuracies <- append(accuracies, acc)
    }
    
  }
  
  return (list(weights=weights, accuracies=accuracies))
  
}

nn <- main_loop()

# Plot to show the error through epochs.
plot(nn$accuracies, type="l", xlab="Number of epochs in 100 epchs per unit", ylab="Error", main="Test error on MNIST perceptron through epochs")







