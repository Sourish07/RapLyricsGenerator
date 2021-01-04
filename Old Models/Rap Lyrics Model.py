import time

import tensorflow as tf

import numpy as np
import os


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True,
                                recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dropout(rate=0.2),
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

    return start_string + ''.join(text_generated)


if __name__ == '__main__':
    # Parameters for program
    name = "Travis Scott"
    artist_name = name
    EPOCHS = 30
    temperature = 1.0  # Low means more predictable results, high means less
    num_chars_generate = 300
    start_string = u"Let's "
    train_model = False

    # Don't really see need to change
    BATCH_SIZE = 16
    BUFFER_SIZE = 10000
    embedding_dim = 256
    rnn_units = 256

    name = name.lower().replace(" ", "_")
    # filename = f'{name}_training_data.txt'
    # filename = f'clean_{name}_training_data.txt'
    filename = 'duplicated_travis_scott_data.txt'
    # filename = 'butterfly_effect.txt'
    text = open(filename, 'r', encoding='utf-8').read()
    # print('Length of text: {} characters'.format(len(text)))

    unique_chars = sorted(set(text))
    # print(f"Number of unique characters: {len(vocab)}")

    chars_to_ids = {u: i for i, u in enumerate(unique_chars)}
    ids_to_chars = np.array(unique_chars)

    text_as_int = np.array([chars_to_ids[c] for c in text])

    segment_length = 25
    examples_per_epoch = len(text) // (segment_length + 1)

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(segment_length + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    vocab_size = len(unique_chars)

    model = build_model(vocab_size=len(unique_chars), embedding_dim=embedding_dim, rnn_units=rnn_units,
                        batch_size=BATCH_SIZE)
    model.compile(optimizer='adam', loss=loss)

    checkpoint_dir = f'{name}_training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

    if train_model:
        history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    # Restoring the last checkpoint
    tf.train.latest_checkpoint(checkpoint_dir)
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))

    start = time.time()
    print(f"Generating lyrics like {artist_name}")
    print(generate_text(model, start_string=start_string, num_chars_generate=num_chars_generate,
                        temperature=temperature))
    end = time.time()
    print(f"Took {end - start} seconds to generate")
