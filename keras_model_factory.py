from keras.models import Sequential
from keras.layers import Dense, LSTM
from layers import att_time_cls


def create_lstm_model(batch_size, num_hidden_units, num_steps, num_features, num_classes):
    model = Sequential()
    model.add(LSTM(num_hidden_units, input_shape=(num_steps, num_features), return_sequences=False, stateful=False))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_attention_time_model(batch_size, num_hidden_units, num_steps, num_features, num_classes):
    model = Sequential()
    model.add(LSTM(num_hidden_units, input_shape=(num_steps, num_features), batch_size=batch_size, return_sequences=True, stateful=False))
    model.add(att_time_cls.Attention([batch_size, num_hidden_units], batch_size=batch_size))
    model.add(Dense(num_classes, batch_size=batch_size, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model