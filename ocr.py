from PIL import Image, ImageFilter

TEST_LABEL_FILE = './data/t10k-labels-idx1-ubyte'
TEST_IMAGE_FILE = './data/t10k-images-idx3-ubyte'
TRAIN_LABEL_FILE = './data/train-labels-idx1-ubyte'
TRAIN_IMAGE_FILE = './data/train-images-idx3-ubyte'

DRAWN_TEST_IMAGE = './data/test_image.png'


def convert_bytes_to_int(bytes):
  return int.from_bytes(bytes, 'big')


# Read image bytes and place them into a list to be returned
def scan_images(filename, max_images):
  images = []
  with open(filename, 'rb') as file:
    _ = file.read(4) # file.read(n) will read the next n bytes
    num_of_images = min(convert_bytes_to_int(file.read(4)), max_images)
    num_of_rows = convert_bytes_to_int(file.read(4))
    num_of_cols = convert_bytes_to_int(file.read(4))

    for _ in range(num_of_images):
      image = []
      for _ in range(num_of_rows):
        row = []
        for _ in range(num_of_cols):
          # Read pixel by pixel until we reach our limits
          pixel = file.read(1)
          row.append(pixel)
        image.append(row)
      images.append(image)
    return images

# Read label bytes and place them into a list to be returned
def scan_labels(filename, max_labels):
  labels = []
  with open(filename, 'rb') as file:
    _ = file.read(4)
    num_of_items = min(convert_bytes_to_int(file.read(4)), max_labels)
    for _ in range(num_of_items):
      label = convert_bytes_to_int(file.read(1))
      labels.append(label)
  return labels


# Extract features from our data, in our case flatten the 2D list
def format_list(data):
  flat_list = []
  for image in data:
    flat_list.append([pixel for row in image for pixel in row])
  return flat_list


# Calculate the euclidean distance between the training data and the test data
def euclidean_distance(a, b):
  zipped_values = []
  for cur_a, cur_b in zip(a, b):
    if type(cur_b) == int:
      zipped_values.append((convert_bytes_to_int(cur_a) - cur_b)**2)
    else:
      zipped_values.append((convert_bytes_to_int(cur_a) - convert_bytes_to_int(cur_b))**2)
  return sum(zipped_values)**0.5


# Determine which element appeared most frequently in the KNN
def get_most_frequent_element(candidates):
  return max(set(candidates), key=candidates.count)


# Calculate the euclidean distance for the entire training dataset + test cases
def calculate_distances_for_test(training_dataset, test):
  distances = []
  for data_point in training_dataset:
    distances.append(euclidean_distance(data_point, test))
  return distances


# Our KNN algorithm, this can be subbed out for any other OCR algorithm
def knn(training_dataset, training_labels, test_dataset, k):
  predictions = []
  for index, test in enumerate(test_dataset):
    training_distances = calculate_distances_for_test(training_dataset, test)
    sorted_distances_tuples = sorted(enumerate(training_distances), key=lambda x: x[1])
    
    sorted_distances_indices = []
    for distance in sorted_distances_tuples:
      sorted_distances_indices.append(distance[0])

    candidates = []
    for index2 in sorted_distances_indices[:k]:
      candidates.append(training_labels[index2])

    print("Index: {} ----- Candidates: {}".format(index, candidates))
    top_candidate = get_most_frequent_element(candidates)
    print(" Guess: {}\n".format(top_candidate))
    predictions.append(top_candidate)
  return predictions, candidates


# Determine how accurate our predictions were vs. the actual labels
def check_accuracy(predictions, true_labels):
  correct = 0
  total = len(predictions)
  for cur_prediction, cur_true_label in zip(predictions, true_labels):
    if cur_prediction == cur_true_label:
      correct += 1
  return ((correct / total) * 100)


# Takes a drawn image and will return the list of values for each pixel
def prepare_drawn_image(image):
  # Create image with black canvas (28 x 28)
  # image = Image.open(path).convert('L')
  width = float(image.size[0])
  height = float(image.size[1])
  print("W", width)
  print("H", height)
  canvas = Image.new('L', (70, 70), (0))

  # If image is not a sqaure, resize to fit width or height of 20 pixels
  nheight = int(round((50.0 / width * height), 0))
  print("NH", nheight)
  if (nheight == 0):
    nheight = 1

  resized_image = image.resize((35, 35), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
  canvas.paste(resized_image, (7, 7))

  # Save the image for test viewing and return the list of values for calculation
  final_canvas = canvas.crop((11, 11, 39, 39))
  final_canvas.save("./data/drawn_image.png")
  print(canvas.size)
  return list(final_canvas.getdata())
  
  # canvas.save("./data/drawn_image.png")
  # return list(canvas.getdata())



# def main():
#   print('Chosen values: TRAIN_VALUES={}, TEST_VALUES={}, K={}\n'.format(TRAIN_VALUES, TEST_VALUES, K))
#   training_dataset = scan_images(TRAIN_IMAGE_FILE, TRAIN_VALUES)
#   training_labels = scan_labels(TRAIN_LABEL_FILE, TRAIN_VALUES)
#   test_dataset = scan_images(TEST_IMAGE_FILE, TEST_VALUES)
#   test_labels = scan_labels(TEST_LABEL_FILE, TEST_VALUES)


#   # MY TEST
#   # flat_test_dataset = [prepare_drawn_image(DRAWN_TEST_IMAGE)]
#   # test_labels = [1]

#   flat_training_dataset = format_list(training_dataset)
#   flat_test_dataset = format_list(test_dataset)

#   predictions = knn(flat_training_dataset, training_labels, flat_test_dataset, K)

#   print("Guesses: {}".format(predictions))

#   accuracy = check_accuracy(predictions, test_labels)
#   print("Accuracy: {}%".format(accuracy))