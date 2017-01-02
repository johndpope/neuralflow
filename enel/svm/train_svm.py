import numpy as np
from enel.Dataset import Dataset
from enel.EnelDataset import EnelDataset
from enel.HeldOutEvaluationMethod import HeldOutEvaluationMethod
from enel.ModelEvaluation import ModelEvaluation
from enel.SVR import SVRTrainingStrategy
from enel.svm.Metric import MAE

root_dir = "/home/giulio/datasets/enel_mats/"

data = EnelDataset(mat_file=root_dir + "/SUD_by_hour_preprocessed.mat", seed=12,
                   scale_y=True, name="SUD")

models = [SVRTrainingStrategy(C=c, gamma=g, epsilon=e)
          for c in np.linspace(0.1, 1, 10)
          for g in np.logspace(-8, -1, 20)
          for e in np.linspace(0.01, 0.2, 10)]

validation_set = Dataset(data.get_validation()["input"], data.get_validation()["output"])
train_set = Dataset(data.get_train()["input"], data.get_train()["output"])

score_fnc = MAE(y_scaler=data.y_scaler)

evaluation_method = HeldOutEvaluationMethod(train_set=train_set, validation_set=validation_set,
                                            score_fnc=score_fnc)

model_eval = ModelEvaluation(method=evaluation_method, n_threads=8)
configurations = model_eval.run(models)

best_config = sorted(configurations,
                       key=lambda x: x["score"].item())[-1]

print("Best config: {} -> validation score: {:.2f}".format(str(best_config["strategy"]), best_config["score"]))

combined_train = train_set.combine(validation_set)
model = best_config["strategy"].train(training_set=combined_train)

test_set = Dataset(data.get_test()["input"], data.get_test()["output"])

predictions = model.predict(test_set.X)

error = score_fnc.compute_score(predictions, test_set.Y)

print("Test error: {:.2f}".format(error))
