import idx2numpy
import numpy as np

def filter_classes(images, labels):
    # Mask untuk menyaring label yang nilainya kurang dari atau sama dengan 35
    mask = labels <= 35

    filtered_images = images[mask]
    filtered_labels = labels[mask]

    return filtered_images, filtered_labels


def load_data():
    # Membaca data train
    train_images = idx2numpy.convert_from_file(
        'data/emnist-byclass-train-images-idx3-ubyte'
    )

    train_labels = idx2numpy.convert_from_file(
        'data/emnist-byclass-train-labels-idx1-ubyte'
    )

    # Membaca data test
    test_images = idx2numpy.convert_from_file(
        'data/emnist-byclass-test-images-idx3-ubyte'
    )

    test_labels = idx2numpy.convert_from_file(
        'data/emnist-byclass-test-labels-idx1-ubyte'
    )

    # Filter train
    train_images, train_labels = filter_classes(
        train_images,
        train_labels
    )

    # Filter test
    test_images, test_labels = filter_classes(
        test_images,
        test_labels
    )

    return (
        train_images,
        train_labels,
        test_images,
        test_labels
    )


def vectorized_result(j):
    """
    One-hot encoding untuk 36 kelas
    """
    e = np.zeros((36, 1))
    e[j] = 1.0
    return e


def load_data_wrapper():
    train_images, train_labels, test_images, test_labels = load_data()

    # =========================
    # TRAINING DATA
    # =========================

    training_inputs = []

    for img in train_images:

        # PERBAIKAN EMNIST:
        # rotate + flip agar orientasi sama dengan gambar dari canvas
        img = np.transpose(img)

        img = img.astype(np.float32) / 255.0

        training_inputs.append(
            np.reshape(img, (784, 1))
        )

    training_results = [
        vectorized_result(label)
        for label in train_labels
    ]

    training_data = list(
        zip(training_inputs, training_results)
    )

    # =========================
    # TEST DATA
    # =========================

    test_inputs = []

    for img in test_images:

        # PERBAIKAN EMNIST
        img = np.transpose(img)

        img = img.astype(np.float32) / 255.0

        test_inputs.append(
            np.reshape(img, (784, 1))
        )

    test_data = list(
        zip(test_inputs, test_labels)
    )

    return training_data, None, test_data