import time

import tensorflow as tf

import numpy as np
import os


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_model(vocab_size, embedding_dim, rnn_units, batch_size, dropout_rate):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.Dropout(rate=dropout_rate),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(rate=dropout_rate),
        tf.keras.layers.Dense(vocab_size)
    ])
    # return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def generate_text(model, start_string, num_chars_generate, temperature):
    input_eval = [chars_to_ids[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    model.reset_states()
    for _ in range(num_chars_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(ids_to_chars[predicted_id])

    return ''.join(text_generated)
    # return start_string + ''.join(text_generated)


if __name__ == '__main__':
    # Parameters for program
    name = "Travis Scott"
    artist_name = name
    EPOCHS = 30
    temperature = 0.9  # Low means more predictable results, high means less
    num_chars_generate = 300
    learning_rate = 0.01
    dropout_rate = 0.2

    BATCH_SIZE = 16
    BUFFER_SIZE = 10000
    embedding_dim = 256
    rnn_units = 256
    segment_length = 40
    segment_shift = 1

    start_string = u"Let's "
    train_model = False

    name = name.lower().replace(" ", "_")
    # filename = f'{name}_training_data.txt'
    # filename = f'clean_{name}_training_data.txt'
    filename = f'duplicated_travis_scott_data.txt'
    # filename = 'butterfly_effect.txt'
    text = open(filename, 'r', encoding='utf-8').read()
    # print('Length of text: {} characters'.format(len(text)))

    unique_chars = sorted(set(text))
    # print(f"Number of unique characters: {len(vocab)}")

    chars_to_ids = {u: i for i, u in enumerate(unique_chars)}
    ids_to_chars = np.array(unique_chars)

    text_as_int = np.array([chars_to_ids[c] for c in text])

    examples_per_epoch = len(text) // (segment_length + 1)

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    dataset = char_dataset.window(segment_length + 1, shift=segment_shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(segment_length + 1))

    # sequences = char_dataset.batch(segment_length + 1, drop_remainder=True)

    dataset = dataset.map(split_input_target)
    print("DATASET: ")
    print(dataset)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    vocab_size = len(unique_chars)

    model = build_model(vocab_size=len(unique_chars), embedding_dim=embedding_dim, rnn_units=rnn_units,
                        batch_size=BATCH_SIZE, dropout_rate=dropout_rate)

    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=adam_optimizer, loss=loss)

    checkpoint_dir = f'{name}_training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

    if train_model:
        history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    # Restoring the last checkpoint
    tf.train.latest_checkpoint(checkpoint_dir)
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1, dropout_rate=dropout_rate)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))

    start = time.time()
    print(f"Generating lyrics like {artist_name}")
    print(generate_text(model, start_string=start_string, num_chars_generate=num_chars_generate,
                        temperature=temperature))
    end = time.time()
    print(f"Took {end - start} seconds to generate")
