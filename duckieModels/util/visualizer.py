import pickle

import imageio
from matplotlib import cm
from vis.visualization import visualize_cam

from duckieLog.log_util import read_dataset
from duckieModels.cbcNet import cbcNet
from duckieModels.cbcNetv2 import cbcNetv2
from duckieModels.util.salient_visualizer import *


def get_model_summary(model):
    print(model.summary())
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_dtype=True,
        show_layer_names=True, rankdir='TB', expand_nested=True, dpi=128
    )


def load_anomaly_example(anomaly):
    if anomaly:
        file = open("../tests/anomaly_img", 'rb')
    else:
        file = open("../tests/normal_img", 'rb')
    imgs = pickle.load(file)
    file.close()
    return imgs


def anomaly_naive_test(model):
    imgs = load_anomaly_example(anomaly=True)
    for an_img in imgs:
        observation = np.expand_dims(an_img, axis=0)
        anomaly = model.predict(observation)
        print("Anomaly={}, GT=True, at {}".format(anomaly[0][0] > 0.5, round(anomaly[0][0], 2)))

    imgs = load_anomaly_example(anomaly=False)
    for an_img in imgs:
        observation = np.expand_dims(an_img, axis=0)
        anomaly = model.predict(observation)
        print(anomaly)
        print("Anomaly={}, GT=False, at {}".format(anomaly[0][0] > 0.5, round(anomaly[0][0], 2)))


def anomaly_detection_attention(model, logdir="../tests/ducks.log"):
    model = cbcNetv2.get_anomaly_inference(model)
    observation, prediction, anomaly = read_dataset(
        logdir
    )
    writer = imageio.get_writer("../TrainingDataVisualization.mp4", format='mp4', mode='I', fps=60.0)
    for a_observation, a_gt in zip(observation, anomaly):
        observation = np.expand_dims(a_observation, axis=0)
        output = model.predict(observation)
        heatmap = visualize_cam(model, layer_idx=-1, filter_indices=0,
                                seed_input=a_observation, grad_modifier=None)
        jet_heatmap = cv2.cvtColor(np.uint8(cm.jet(heatmap)[..., :3] * 255), cv2.COLOR_BGR2RGB)
        plt_img = cv2.cvtColor(a_observation, cv2.COLOR_YUV2RGB)
        fin = cv2.addWeighted(plt_img, 0.6, jet_heatmap, 0.4, 0)
        string_to_show = "Guess: {}, GT: {}".format(round(float(output[0][0]), 2), a_gt)
        x, y, w, h = 0, 0, 30, 30
        # Add text
        cv2.putText(fin, string_to_show, (x + int(w / 10), y + int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)
        cv2.imshow("Output", fin)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        writer.append_data(cv2.cvtColor(fin, cv2.COLOR_RGB2BGR))
    writer.close()


def salient_visualizer_v1():
    v1_model = cbcNet.get_inference("../cbcNet.h5")
    observation, prediction, anomaly = read_dataset(
        "../tests/ducks.log"
    )
    for frame in observation:
        salient_map_mean, action_dist_params = nvidia_salient_map(v1_model, frame)
        display_salient_map(salient_map_mean, frame, "Salient objects for action mean")


if __name__ == '__main__':
    # COVVisualizerNode = COVVisualizer("cbcNet.h5", "../tests/curve.jpg")
    # COVVisualizerNode.visualize()
    # get_model_summary()
    # imgs = load_anomaly_example(True)
    # anomaly_detection_attention("../cbcNetv2_anomaly.h5")
    # BC_detection_attention("../cbcNetv2_bc.h5")
    salient_visualizer_v1()
