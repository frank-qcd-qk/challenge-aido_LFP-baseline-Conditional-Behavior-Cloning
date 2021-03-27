import tensorflow as tf
from tensorflow.keras.backend import less_equal as Less_equal
from tensorflow.keras.layers import Conv2D, Lambda, Flatten, Dense


class cbcNetv2:
    @staticmethod
    def build_cbc_anomaly_detector(rgb_image):
        # ? Input Normalization
        normalized_image = Lambda(lambda x: x / 255.0)(rgb_image)
        # ? Anomaly Detector:
        # ? L1: CONV => RELU
        anomaly_branch = Conv2D(24, (5, 5), strides=(2, 2), activation='relu', padding="valid", name='AN_Conv1')(
            normalized_image)
        # ? L2: CONV => RELU
        anomaly_branch = Conv2D(36, (5, 5), strides=(2, 2), activation='relu', padding="valid", name='AN_Conv2')(
            anomaly_branch)
        # ? L3: CONV => RELU
        anomaly_branch = Conv2D(64, (3, 3), activation='relu', padding="valid", name='AN_Conv3')(anomaly_branch)
        # ? Flatten
        anomaly_branch = Flatten()(anomaly_branch)
        # ? Fully Connected
        anomaly_branch = Dense(100, kernel_initializer='normal', activation='relu', name='AN_FC1')(anomaly_branch)
        anomaly_branch = Dense(50, kernel_initializer='normal', activation='relu', name='AN_FC2')(anomaly_branch)
        anomaly_branch = Dense(10, kernel_initializer='normal', activation='relu', name='AN_FC3')(anomaly_branch)
        anomaly = Dense(1, kernel_initializer='normal', activation='sigmoid', name="Anomaly_Out")(anomaly_branch)
        return anomaly

    @staticmethod
    def build_cbc_net(rgb_image, anomaly_inject):
        # ? Input Normalization
        normalized_image = Lambda(lambda x: x / 255.0)(rgb_image)
        # ? Behavior Cloning:
        # ? L1: CONV => RELU
        bc_branch = Conv2D(24, (5, 5), strides=(2, 2), padding="valid", activation='relu', name='BC_Conv1')(
            normalized_image)
        # ? L2: CONV => RELU
        bc_branch = Conv2D(36, (5, 5), strides=(2, 2), activation='relu', padding="valid", name='BC_Conv2')(bc_branch)
        # ? L3: CONV => RELU
        bc_branch = Conv2D(48, (5, 5), strides=(2, 2), activation='relu', padding="valid", name='BC_Conv3')(bc_branch)
        # ? L4: CONV => RELU
        bc_branch = Conv2D(64, (3, 3), activation='relu', padding="valid", name='BC_Conv4')(bc_branch)
        # ? L5: CONV => RELU
        bc_branch = Conv2D(64, (3, 3), activation='relu', padding="valid", name='BC_Conv5')(bc_branch)
        # ? Flatten
        bc_branch = Flatten()(bc_branch)

        # ? Initial Fully Connected
        prediction = Dense(1164, kernel_initializer='normal', activation='relu', name='BC_FC1')(bc_branch)

        x = Dense(500, kernel_initializer='normal', activation='relu', name='ANB_FC1')(prediction)
        x = Dense(50, kernel_initializer='normal', activation='relu', name='ANB_FC2')(x)
        x = Dense(10, kernel_initializer='normal', activation='relu', name='ANB_FC3')(x)
        x = Dense(2, kernel_initializer='normal', name='ANB_Out')(x)

        y = Dense(500, kernel_initializer='normal', activation='relu', name='BCB_FC1')(prediction)
        y = Dense(50, kernel_initializer='normal', activation='relu', name='BCB_FC2')(y)
        y = Dense(10, kernel_initializer='normal', activation='relu', name='BCB_FC3')(y)
        y = Dense(2, kernel_initializer='normal', name='BCB_Out')(y)

        # ? Switch
        prediction = tf.where(Less_equal(anomaly_inject, 0.5), x, y, name="Prediction")
        return prediction

    @staticmethod
    def get_model(lr, epochs, input_shape=(150, 200, 3)):
        # ! Define input
        rgb_input = tf.keras.Input(shape=input_shape)
        anomaly_input = tf.keras.Input(shape=(1))

        # ! Build Structure
        driving_cmd = cbcNetv2.build_cbc_net(rgb_input, anomaly_input)
        anomaly_output = cbcNetv2.build_cbc_anomaly_detector(rgb_input)
        cmd_model = tf.keras.Model(inputs=[rgb_input, anomaly_input], outputs=driving_cmd, name="cbcNet")
        anomaly_model = tf.keras.Model(inputs=rgb_input, outputs=anomaly_output, name="cbcNet_anomaly")
        # ! Setup Optimizer
        opt = tf.keras.optimizers.Adam(lr=lr, decay=lr / epochs)
        # ! Compile Model
        cmd_model.compile(
            optimizer=opt, loss="mse", metrics="mse"
        )
        anomaly_model.compile(
            optimizer=opt, loss="binary_crossentropy", metrics=['binary_accuracy', 'binary_crossentropy']
        )
        return cmd_model, anomaly_model

    @staticmethod
    def get_anomaly_inference(weigths="cbcNetv2_anomaly.h5", input_shape=(150, 200, 3)):
        rgb_input = tf.keras.Input(shape=input_shape)
        anomaly_detection = cbcNetv2.build_cbc_anomaly_detector(rgb_input)
        model = tf.keras.Model(inputs=rgb_input, outputs=anomaly_detection, name="cbcNet-anomaly")
        model.load_weights(weigths)
        return model

    @staticmethod
    def get_bc_inference(weigths="cbcNet_bc.h5", input_shape=(150, 200, 3)):
        rgb_input = tf.keras.Input(shape=input_shape)
        anomaly_input = tf.keras.Input(shape=(1))
        bc_output = cbcNetv2.build_cbc_net(rgb_input, anomaly_input)
        model = tf.keras.Model(inputs=[rgb_input, anomaly_input], outputs=bc_output, name="cbcNet-bc")
        model.load_weights(weigths)
        return model
