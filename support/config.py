import os

# data params
str_len = 20
str_classes = str_len + 1
time_stamps = str_len
feature_len = 1

# model params
lstm_len = str_len
batch_size = 128
epochs = 10
validation_split = 0.2
input_shape = (time_stamps, feature_len)
output_shape = str_classes
optimizer = 'adam'
loss = 'categorical_crossentropy'
metrics = ['acc']

model_folder = r'./model_archive'
model_name = 'count_of_1s_' + str(c.str_len) + '.h5'
model_path = os.path.join(c.model_path, model_name)

# loss plot
plot_loss = False
