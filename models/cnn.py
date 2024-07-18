import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


class PneumoniaClassifier:

    def __init__(self, train_dir, val_dir, test_dir, image_size=(128, 128), batch_size=32):
        self.train_dir = train_dir,
        self.val_dir = val_dir,
        self.test_dir = test_dir,
        self.image_size = image_size
        self.batch_size = batch_size
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_classes = None
        self.model = None
        self.history = None

    def load_data(self, data_dir):
        return tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            color_mode='grayscale',
            image_size=(128, 128),
            batch_size=32
        )

    def display_image(self, dataset):
        plt.figure(figsize=(10, 10))
        for images, labels in dataset.take(1):
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                plt.imshow(np.squeeze(images[i].numpy().astype('uint8')), cmap='gray')
                plt.title(dataset.class_names[labels[i]])
                plt.axis('off')

    def plot_distribution(self, dataset, title):
        labels = []
        for images, lbls in dataset:
            labels.extend(lbls.numpy())

        labels_count = {label: labels.count(label) for label in set(labels)}
        colors = ['skyblue', 'lightcoral']

        plt.bar(labels_count.keys(), labels_count.values(), color=colors)
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title(f'Distribution of PNEUMONIA and NORMAL in {title} Dataset')
        plt.xticks(ticks=[0, 1], labels=["NORMAL", "PNEUMONIA"])
        plt.show()

    def dataset_to_numpy(self, dataset):
        images = []
        labels = []

        for batch in dataset:
            batch_images, batch_labels = batch
            images.extend(batch_images.numpy())
            labels.extend(batch_labels.numpy())

        images = np.array(images)
        labels = np.array(labels)

        return images, labels

    def prepare_data(self, test_size=0.2, random_state=42):
        train = self.load_data(self.train_dir[0])
        val = self.load_data(self.val_dir[0])
        test = self.load_data(self.test_dir[0])

        train_images, train_labels = self.dataset_to_numpy(train)
        val_images, val_labels = self.dataset_to_numpy(val)
        test_images, test_labels = self.dataset_to_numpy(test)

        x = np.concatenate((train_images, val_images, test_images), axis=0)
        y = np.concatenate((train_labels, val_labels, test_labels), axis=0)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, random_state=random_state,
                                                                                test_size=test_size)

    def build_model(self, metrics):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1))),
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.summary()
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=metrics)

        self.model = model

    def train_model(self, epochs=10, validation_split=0.3):
        self.y_train = self.y_train.reshape(-1, 1)
        self.y_test = self.y_test.reshape(-1, 1)

        self.history = self.model.fit(self.X_train, self.y_train,
                                      batch_size=self.batch_size,
                                      epochs=epochs,
                                      validation_split=validation_split)

    def evaluate_model(self):
        return self.model.evaluate(self.X_test, self.y_test, verbose=2)

    def classification_report(self, threshold=0.5):
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = (y_pred > threshold).astype("int32")
        print(classification_report(self.y_test, y_pred_classes))
        return y_pred, y_pred_classes

    def confusion_matrix(self, y_pred_classes):
        conf = confusion_matrix(self.y_test, y_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matrice de Confusion")
        plt.show()

    def save_model(self, path):
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")

