from keras import Model
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, RNN, Embedding, Concatenate, Multiply
from layers import att_time_cls, att_input_rnn


def create_lstm_model(batch_size, num_hidden_units, num_steps, num_features, num_classes):
    model = Sequential()
    model.add(LSTM(num_hidden_units, input_shape=(num_steps, num_features), return_sequences=False, stateful=False))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_attention_time_model(batch_size, num_hidden_units, num_steps, num_features, num_classes):
    model = Sequential()
    model.add(
        LSTM(num_hidden_units, input_shape=(num_steps, num_features), batch_size=batch_size, return_sequences=True,
             stateful=False))
    model.add(att_time_cls.Attention([batch_size, num_hidden_units], batch_size=batch_size))
    model.add(Dense(num_classes, batch_size=batch_size, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_attention_time_continuous_model(batch_size, num_hidden_units, num_steps, num_features, num_classes):
    model = Sequential()
    model.add(
        LSTM(num_hidden_units, input_shape=(num_steps, num_features), batch_size=batch_size, return_sequences=True,
             stateful=False))
    model.add(att_time_cls.Attention([batch_size, num_hidden_units], batch_size=batch_size, continuous=True))
    model.add(Dense(num_classes, batch_size=batch_size, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_attention_input_rnn_model(batch_size, num_hidden_units, num_steps, num_features, num_classes):
    main_input = Input(batch_shape=(batch_size, num_steps, num_features), name='main_input')
    h = LSTM(num_hidden_units, input_shape=(num_steps, num_features), batch_size=batch_size, return_sequences=True)(main_input)

    att = att_input_rnn.Attention(n_feature=num_features, n_sensor=3, batch_size=batch_size)(h)
    x = Multiply()([main_input, att])
    lstm_out = LSTM(num_hidden_units, input_shape=(num_steps, num_features), batch_size=batch_size,
                    return_sequences=False)(x)
    out = Dense(num_classes, batch_size=batch_size, activation="softmax")(lstm_out)
    model = Model(inputs=[main_input], outputs=[out])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
