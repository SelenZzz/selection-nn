from genericpath import isdir
import os
from utils.rangedict import RangeDict
import utils.nn_tools as nn_tools

if not os.path.exists("./plots"):
    os.mkdir("./plots")

input_shape = [1]


def summary_model():
    # summary
    n_ins = 10
    n_outs = 4
    range_dict = RangeDict(
        {
            range(0, 3): 0,
            range(3, 5): 1,
            range(5, 8): 2,
            range(8, 10): 3,
        }
    )
    model_path = "./saved_models/summary_model.h5"
    x, y = nn_tools.generate_data(n_ins, n_outs, range_dict)
    if os.path.isfile(model_path):
        model = nn_tools.load(model_path)
    else:
        model = nn_tools.build_model(n_ins, input_shape, n_outs)
        model = nn_tools.train(x, y, model, model_path=model_path, plot_name="summary")
    return model


def weak_model():
    # weak
    n_ins = 10
    n_outs = 2
    range_dict = RangeDict(
        {
            range(0, 5): 0,
            range(5, 10): 1,
        }
    )
    model_path = "./saved_models/weak_model.h5"
    x, y = nn_tools.generate_data(n_ins, n_outs, range_dict)
    if os.path.isfile(model_path):
        model = nn_tools.load(model_path)
    else:
        model = nn_tools.build_model(n_ins, input_shape, n_outs)
        model = nn_tools.train(x, y, model, model_path=model_path, plot_name="weak")
    return model
