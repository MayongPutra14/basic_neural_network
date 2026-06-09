import emnist_loader

training_data, _, test_data = emnist_loader.load_data_wrapper()

print(len(training_data))
print(len(test_data))

x, y = training_data[0]

print(x.shape)
print(y.shape)