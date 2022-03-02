from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from functions.util import *


class STLearner:

    def __init__(self, model=LinearRegression, params=None, quartile=True):

        if params is not None:
            self.model = model(**params)
        else:
            self.model = model()

        self.unique_treatment = None
        self.quartile = quartile

    def _batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def fit(self, x, y, t):

        unique_treatment = np.unique(t)
        unique_treatment = (unique_treatment[1:] + unique_treatment[:-1]) / 2
        unique_treatment = unique_treatment[1:-1]

        if self.quartile:
            first_quartile = int(np.floor(unique_treatment.shape[0] / 4))
            third_quartile = int(np.ceil(3 * unique_treatment.shape[0] / 4))

            unique_treatment = unique_treatment[first_quartile:third_quartile]

        self.unique_treatment = unique_treatment

        if len(t.shape) < 2:
            fit_x = np.hstack((x, t.reshape(-1, 1)))
        else:
            fit_x = np.hstack((x, t))

        self.model.fit(fit_x, y)

    def triggers(self, x, batch_size=None, verbose=False):
        indices = np.arange(x.shape[0])
        if batch_size is None:
            batch = len(x)
        else:
            batch = batch_size
        if self.unique_treatment.shape[0] < 2 or x.shape[0] < 1:
            return np.zeros(x.shape[0])
        full_predictions = []
        for i, subset_idx in enumerate(self._batch(indices, batch)):
            if verbose:
                print(f"Batch {i}")
            temp_x = x[subset_idx, :]
            repeat_x = np.repeat(temp_x, self.unique_treatment.shape[0], axis=0)
            repeat_t = np.tile(self.unique_treatment, temp_x.shape[0]).reshape(-1, 1)
            full_repeat = np.hstack((repeat_x, repeat_t))

            # print(full_repeat.shape)

            full_pred = self.model.predict(full_repeat)
            num_t = self.unique_treatment.shape[0]
            trigger_indices = []
            for i in range(num_t, full_pred.shape[0] + num_t, num_t):
                test = full_pred[i - num_t:i]
                effects = np.array([np.mean(test[j:]) - np.mean(test[:j]) for j in range(1, test.shape[0])])
                trigger_indices.append(np.argmax(effects))
            trigger_indices = np.array(trigger_indices)
            pred_triggers = self.unique_treatment[trigger_indices]
            full_predictions.append(pred_triggers)
        full_predictions = np.concatenate(full_predictions)
        return full_predictions
        # repeat_x = np.repeat(x[subset_idx, :], self.unique_treatment.shape[0], axis=0)
        # repeat_t = np.tile(self.unique_treatment, x.shape[0]).reshape(-1, 1)
        # full_repeat = np.hstack((repeat_x, repeat_t))
        #
        # if self.unique_treatment.shape[0] < 2 or x.shape[0] < 1:
        #     return np.zeros(x.shape[0])
        #
        # full_pred = self.model.predict(full_repeat)
        # num_t = self.unique_treatment.shape[0]
        # trigger_indices = []
        # for i in range(num_t, full_pred.shape[0] + num_t, num_t):
        #     test = full_pred[i - num_t:i]
        #     effects = np.array([np.mean(test[j:]) - np.mean(test[:j]) for j in range(1, test.shape[0])])
        #     trigger_indices.append(np.argmax(effects))
        # trigger_indices = np.array(trigger_indices)
        # pred_triggers = self.unique_treatment[trigger_indices]
        # return pred_triggers

    def _predict_outcome(self, x):
        if len(x.shape) < 2:
            self.model.predict(x.reshape(-1, 1))
        else:
            self.model.predict(x)
