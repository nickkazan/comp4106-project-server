import flask
import base64
import io
from PIL import Image
from ocr import prepare_drawn_image, scan_images, scan_labels, format_list, knn, check_accuracy

application = flask.Flask(__name__)

TRAIN_LABEL_FILE = './data/train-labels-idx1-ubyte'
TRAIN_IMAGE_FILE = './data/train-images-idx3-ubyte'


@application.route('/process-image', methods=['POST'])
def process_image():
  args = flask.request.get_json()
  train_values = int(args['train_values'])
  test_values = int(args['test_values'])
  k = int(args['k'])
  image_data = args['image']
  image_b64 = image_data[22:]
  image_binary = base64.b64decode(image_b64)
  buffer = io.BytesIO(image_binary)
  image = Image.open(buffer).convert("L")

  flat_test_dataset = [prepare_drawn_image(image)]

  print('Chosen values: TRAIN_VALUES={}, TEST_VALUES={}, K={}\n'.format(train_values, test_values, k))

  if flask.request.files and flask.request.files['image']:
    file = flask.request.files['image']
    image = Image.open(file.stream).convert('L')
    flat_test_dataset = [prepare_drawn_image(image)]

  training_dataset = scan_images(TRAIN_IMAGE_FILE, train_values)
  training_labels = scan_labels(TRAIN_LABEL_FILE, train_values)
  # test_dataset = scan_images(TEST_IMAGE_FILE, test_values)
  # test_labels = scan_labels(TEST_LABEL_FILE, test_values)

  flat_training_dataset = format_list(training_dataset)
  # flat_test_dataset = format_list(test_dataset)

  predictions, candidates = knn(flat_training_dataset, training_labels, flat_test_dataset, k)

  print("Predictions: {}".format(predictions))
  print("Candidates: {}".format(candidates))

  # add our predictions into the list for returning
  candidates.insert(0, predictions[0])

  return flask.jsonify(candidates)

if __name__ == "__main__":
  application.run(debug=True)