import pickle
from flask import Flask, request
import tensorflow as tf
from main import make_model


string_feature_names = ['Sex', 'Embarked']
model_API = make_model()
model_API.load_weights('./checkpoints/model_checkpoint')

file_handle_1 = open("vocabs.obj", 'rb')
vocabs = pickle.load(file_handle_1)
file_handle_1.close()

file_handle_2 = open("categorical_feature_names.obj", 'rb')
categorical_feature_names = pickle.load(file_handle_2)
file_handle_2.close()

normalizer_model = tf.keras.models.load_model('normalizer_files')
normalizer = normalizer_model.layers[0]

app = Flask(__name__)


@app.get("/countries")
def get_countries():
    new_entry = [normalizer([float(request.args['Age']), float(request.args['Fare'])])][0]
    for name in categorical_feature_names:
        if type(vocabs[name][0]) is str:
            lookup = tf.keras.layers.StringLookup(vocabulary=vocabs[name], output_mode='one_hot')
        else:
            lookup = tf.keras.layers.IntegerLookup(vocabulary=vocabs[name], output_mode='one_hot')
        if name in string_feature_names:
            x_partial_train = lookup(request.args[name])
        else:
            x_partial_train = lookup(int(request.args[name]))
        new_entry = tf.concat([new_entry, tf.reshape(x_partial_train, [1, -1])], 1)
    res = model_API.predict(new_entry) > .5
    if res[0][0]:
        return "Survived"
    else:
        return "Did not survive"


app.run()
