import numpy as np
import pandas as pd
from utils.output import style
from statistics import mean
from utils.nn_tools import to_index

import nn


def classify(classes, model, value):
    pred = model.predict([value])
    index = to_index(pred[0])
    classified = classes[index]
    return classified


model_weak = nn.weak_model()
model_sum = nn.summary_model()

classes = ["Критично", "Некритично"]
df = pd.read_csv("table.csv", index_col=0)
columns = [i.lstrip() for i in df.columns if isinstance(df[i].iloc[1], np.int64)]
means = [mean(df[i].tolist()) for i in columns]
print(style.BOLD + "Найдены метрики (avg):" + style.END)
[print(f"\t{columns[i]}: {means[i]} - {classify(classes,model_weak,[means[i]])}") for i in range(0, len(columns))]

classes = ["Низко", "Средне", "Хорошо", "Отлично"]
mean_summary = mean(means)
classified = classify(classes, model_sum, [mean_summary])
print(f"{style.BOLD}Общая оценка:{style.END} {classified}")
