from utils import *

st.title('Noise Enhancement using noise suppression')

model = tf.keras.models.load_model("Model.h5")
file_uploader = st.sidebar.file_uploader(label="", type=".wav")

st.subheader("")
st.subheader("Input Speech Sample")


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if file_uploader is not None:
    out = predict(file_uploader)

    st.subheader("")
    st.subheader("")

    wav_encoder = tf.audio.encode_wav(out, 16000)

    test = wav_encoder.numpy()

    st.subheader("Output Speech Sample")

    st.audio(test)
