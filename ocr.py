
TEST_LABEL_FILE = './data/t10k-labels-idx1-ubyte'
TEST_IMAGE_FILE = './data/t10k-images-idx3-ubyte'
TRAIN_LABEL_FILE = './data/train-labels-idx1-ubyte'
TRAIN_IMAGE_FILE = './data/train-images-idx3-ubyte'

# These values should be played around with
K = 5
TRAIN_VALUES = 1000
TEST_VALUES = 100


def convert_bytes_to_int(bytes):
  return int.from_bytes(bytes, 'big')

# Read image bytes and place them into a list to be returned
def scan_images(filename, max_images):
  images = []
  with open(filename, 'rb') as file:
    magic_number = file.read(4) # file.read(n) will read the next n bytes
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


def main():
  print('Chosen values: TRAIN_VALUES={}, TEST_VALUES={}, K={}'.format(TRAIN_VALUES, TEST_VALUES, K))
  training_dataset = scan_images(TRAIN_IMAGE_FILE, TRAIN_VALUES)


if __name__ == '__main__':
    main()