import numpy as np
import pydub
import streamlit as st
import tensorflow as tf

from main import model

batching_size = 12000


def handle_uploaded_audio_file(uploaded_file):
    a = pydub.AudioSegment.from_wav(uploaded_file)

    samples = a.get_array_of_samples()

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples.typecode).max

    fp_arr = fp_arr.reshape(fp_arr.shape[0], 1)

    fp_arr = tf.convert_to_tensor(fp_arr, dtype=tf.float32)

    st.write(a)

    return fp_arr


def audio_to_display(audio):
    audio_file = open(audio, 'rb')
    audio_bytes = audio_file.read()
    return audio_bytes


def inference_preprocess(uploaded_file):
    audio = handle_uploaded_audio_file(uploaded_file)
    audio_len = audio.shape[0]
    batches = []
    for i in range(0, audio_len - batching_size, batching_size):
        batches.append(audio[i:i + batching_size])

    batches.append(audio[-batching_size:])
    diff = audio_len - (i + batching_size)
    return tf.stack(batches), diff


def predict(uploaded_file):
    test_data, diff = inference_preprocess(uploaded_file)
    predictions = model.predict(test_data)
    final_op = tf.reshape(predictions[:-1], ((predictions.shape[0] - 1) * predictions.shape[1], 1))
    final_op = tf.concat((final_op, predictions[-1][-diff:]), axis=0)
    return final_op
