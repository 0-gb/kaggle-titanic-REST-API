import pandas as pd
import tensorflow as tf
import pickle


def make_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(29,)))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.01),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    train_df = pd.read_csv('train.csv')
    y_train = train_df.pop('Survived')

    test_df = pd.read_csv('test.csv')
    y_test = pd.read_csv('gender_submission.csv')['Survived']

    numeric_feature_names = [
        'Age',
        'Fare']

    categorical_feature_names = [
        'Pclass',
        'Sex',
        'SibSp',
        'Parch',
        'Embarked']

    numeric_features_train = train_df[numeric_feature_names]
    numeric_features_train = numeric_features_train.fillna(train_df.mean())

    numeric_features_test = test_df[numeric_feature_names]

    categorical_features_train = train_df[categorical_feature_names]
    categorical_features_train = categorical_features_train.apply(lambda x: x.fillna(x.value_counts().index[0]))

    categorical_features_test = test_df[categorical_feature_names]

    numerical_fill_na = {}
    categorical_fill_na = {}
    for name in numeric_feature_names:
        numerical_fill_na[name] = numeric_features_train[name].mean()
        if name in numeric_features_test:
            numeric_features_test[name] = numeric_features_test[name].fillna(numerical_fill_na[name])

    for name in categorical_feature_names:
        categorical_fill_na[name] = categorical_features_train[name].value_counts().index[0]
        if name in categorical_features_test:
            categorical_features_test[name] = categorical_features_test[name].fillna(categorical_fill_na[name])

    preprocessed_data_train = []
    preprocessed_data_test = []
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(numeric_features_train)
    normalizer_model = tf.keras.Sequential([normalizer])
    normalizer_model.compile()
    normalizer_model.save('normalizer_files')

    preprocessed_data_train.append(normalizer(numeric_features_train))
    preprocessed_data_test.append(normalizer(numeric_features_test))
    vocabs = {}
    for name in categorical_feature_names:
        vocabs[name] = sorted(set(categorical_features_train[name]))
        print(f'name: {name}')
        print(f'vocab: {vocabs[name]}\n')

        if type(vocabs[name][0]) is str:
            lookup = tf.keras.layers.StringLookup(vocabulary=vocabs[name], output_mode='one_hot')
        else:
            lookup = tf.keras.layers.IntegerLookup(vocabulary=vocabs[name], output_mode='one_hot')

        x_partial_train = categorical_features_train[name][:, tf.newaxis]
        x_partial_train = lookup(x_partial_train)
        preprocessed_data_train.append(x_partial_train)
        x_partial_test = categorical_features_test[name][:, tf.newaxis]
        x_partial_test = lookup(x_partial_test)
        preprocessed_data_test.append(x_partial_test)
    x_train = preprocessed_data_train[0]
    for element in preprocessed_data_train[1:]:
        x_train = tf.concat([x_train, element], axis=-1)

    x_test = preprocessed_data_test[0]
    for element in preprocessed_data_test[1:]:
        x_test = tf.concat([x_test, element], axis=-1)

    x_train = x_train.numpy()
    model = make_model()
    model.fit(x_train, y_train, epochs=100)
    model.save_weights('./checkpoints/model_checkpoint')

    results = model.evaluate(x_test, y_test)

    file_handle_1 = open("vocabs.obj", "wb")
    pickle.dump(vocabs, file_handle_1)
    file_handle_2 = open("categorical_feature_names.obj", "wb")
    pickle.dump(categorical_feature_names, file_handle_2)

    file_handle_1.close()
    file_handle_2.close()
    #import flask_REST
