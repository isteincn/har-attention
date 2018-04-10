from keras import Model
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, RNN, Embedding, Concatenate, Multiply, Dot, Reshape
from layers import att_time_cls, att_input_rnn, att_input_multihead
from keras import optimizers


def create_lstm_model(batch_size, num_hidden_units, num_steps, num_features, num_classes):
    model = Sequential(name="lstm")
    model.add(LSTM(num_hidden_units, input_shape=(num_steps, num_features), return_sequences=False, stateful=False))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_attention_time_model(batch_size, num_hidden_units, num_steps, num_features, num_classes, lr = 0.001):
    main_input = Input(batch_shape=(batch_size, num_steps, num_features), name='main_input')
    lstm_out = LSTM(num_hidden_units, input_shape=(num_steps, num_features), batch_size=batch_size,
                    return_sequences=True)(main_input)

    att = att_time_cls.Attention([batch_size, num_hidden_units], batch_size=batch_size,
                                 continuous=False, name="att_hidden")(lstm_out)
    ws_lstm = Dot(-2)([lstm_out, att])
    ws_lstm = Reshape([num_hidden_units], input_shape=[num_hidden_units, 1])(ws_lstm)
    out = Dense(num_classes, batch_size=batch_size, activation="softmax")(ws_lstm)
    model = Model(inputs=[main_input], outputs=[out], name="attention_hidden")
    #adam = optimizers.Adam(lr=lr)
    sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
    print('optimizer: sgd')
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def create_attention_time_continuous_model(batch_size, num_hidden_units, num_steps, num_features, num_classes):
    main_input = Input(batch_shape=(batch_size, num_steps, num_features), name='main_input')
    lstm_out = LSTM(num_hidden_units, input_shape=(num_steps, num_features), batch_size=batch_size,
                    return_sequences=True)(main_input)

    att = att_time_cls.Attention([batch_size, num_hidden_units], batch_size=batch_size,
                                 continuous=True, name="att_hidden")(lstm_out)
    ws_lstm = Dot(-2)([lstm_out, att]) # weighted sum of hidden states
    ws_lstm = Reshape([num_hidden_units], input_shape=[num_hidden_units, 1])(ws_lstm)
    out = Dense(num_classes, batch_size=batch_size, activation="softmax")(ws_lstm)
    model = Model(inputs=[main_input], outputs=[out], name="attention_hidden")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_attention_input_rnn_model(batch_size, num_hidden_units, num_steps, num_features, num_classes):
    main_input = Input(batch_shape=(batch_size, num_steps, num_features), name='main_input')
    h = LSTM(num_hidden_units, input_shape=(num_steps, num_features), batch_size=batch_size, return_sequences=True)(main_input)

    att = att_input_rnn.Attention(n_feature=num_features, n_sensor=3, batch_size=batch_size, name="att_input")(h)
    x = Multiply()([main_input, att])
    lstm_out = LSTM(num_hidden_units, input_shape=(num_steps, num_features), batch_size=batch_size,
                    return_sequences=False)(x)
    out = Dense(num_classes, batch_size=batch_size, activation="softmax")(lstm_out)
    model = Model(inputs=[main_input], outputs=[out], name="attention_input_rnn")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_attention_input_rnn_continuous_model(batch_size, num_hidden_units, num_steps, num_features, num_classes):
    main_input = Input(batch_shape=(batch_size, num_steps, num_features), name='main_input')
    h = LSTM(num_hidden_units, input_shape=(num_steps, num_features), batch_size=batch_size, return_sequences=True)(main_input)

    att = att_input_rnn.Attention(n_feature=num_features, n_sensor=3, continuous=True, batch_size=batch_size, name="att_input")(h)
    x = Multiply()([main_input, att])
    lstm_out = LSTM(num_hidden_units, input_shape=(num_steps, num_features), batch_size=batch_size,
                    return_sequences=False)(x)
    out = Dense(num_classes, batch_size=batch_size, activation="softmax")(lstm_out)
    model = Model(inputs=[main_input], outputs=[out], name="attention_input_rnn_continuous")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def create_attention_input_multihead_model(batch_size, num_hidden_units, num_steps, num_features, num_classes, num_heads):
    main_input = Input(batch_shape=(batch_size, num_steps, num_features), name='main_input')
    h = LSTM(num_hidden_units, input_shape=(num_steps, num_features), batch_size=batch_size, return_sequences=True)(main_input)

    att_out = att_input_multihead.Attention(n_feature=num_features, n_sensor=3, n_head=num_heads, batch_size=batch_size)([main_input, h])
    lstm_out = LSTM(num_hidden_units, input_shape=(num_steps, num_heads * 3), batch_size=batch_size,
                    return_sequences=False)(att_out)
    out = Dense(num_classes, batch_size=batch_size, activation="softmax")(lstm_out)
    model = Model(inputs=[main_input], outputs=[out], name="attention_multihead")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# att only multiply without weighted sum, and added L1 L2 regularization
def create_attention_input_rnn_hidden_continuous_model(batch_size, num_hidden_units, num_steps, num_features, num_classes):
    main_input = Input(batch_shape=(batch_size, num_steps, num_features), name='main_input')
    h = LSTM(num_hidden_units, input_shape=(num_steps, num_features), batch_size=batch_size, return_sequences=True)(main_input)
    att = att_input_rnn.Attention(n_feature=num_features, n_sensor=3, continuous=True, batch_size=batch_size, name="att_input")(h)
    x = Multiply()([main_input, att])
    lstm_out = LSTM(num_hidden_units, input_shape=(num_steps, num_features), batch_size=batch_size,
                    return_sequences=True)(x)

    att_hidden = att_time_cls.Attention([batch_size, num_hidden_units], batch_size=batch_size, continuous=True)(lstm_out)
    ws_lstm = Dot(-2)([lstm_out, att_hidden])
    ws_lstm = Reshape([num_hidden_units], input_shape=[num_hidden_units, 1])(ws_lstm)
    out = Dense(num_classes, batch_size=batch_size, activation="softmax")(ws_lstm)
    model = Model(inputs=[main_input], outputs=[out], name="attention_input_rnn_hidden_continuous")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
