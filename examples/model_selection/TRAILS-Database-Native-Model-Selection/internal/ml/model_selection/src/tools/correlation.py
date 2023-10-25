#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from scipy import stats
from src.common.constant import CommonVars
import numpy as np
from src.logger import logger
from sklearn import metrics


class CorCoefficient:

    @staticmethod
    def measure(x1: list, x2: list, measure_metrics: str = CommonVars.AllCorrelation) -> dict:
        """
        Measure the correlation coefficient between x1 and x2
        It requires that each dataset be normally distributed.
        :param x1: list1
        :param x2: list2
        :param measure_metrics: str
        :return: correlation，
            Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation.
            Correlations of -1 or +1 imply an exact linear relationship
        """

        result = {}
        if measure_metrics == CommonVars.KendallTau:
            correlation, p_value = stats.kendalltau(x1, x2, nan_policy='omit')
            result[CommonVars.KendallTau] = correlation
        elif measure_metrics == CommonVars.Spearman:
            correlation, p_value = stats.spearmanr(x1, x2, nan_policy='omit')
            result[CommonVars.Spearman] = correlation
        elif measure_metrics == CommonVars.Pearson:
            correlation, p_value = stats.pearsonr(x1, x2)
            result[CommonVars.Pearson] = correlation
        elif measure_metrics == CommonVars.AvgCorrelation:
            # calculate average over all
            correlation1, p_value = stats.kendalltau(x1, x2, nan_policy='omit')
            correlation2, p_value = stats.spearmanr(x1, x2, nan_policy='omit')
            correlation3, p_value = stats.pearsonr(x1, x2)
            correlation = (correlation1 + correlation2 + correlation3) / 3
            result[CommonVars.AvgCorrelation] = correlation
        elif measure_metrics == CommonVars.AllCorrelation:
            correlation1, p_value = stats.kendalltau(x1, x2, nan_policy='omit')
            correlation2, p_value = stats.spearmanr(x1, x2, nan_policy='omit')
            correlation3, p_value = stats.pearsonr(x1, x2)
            correlation4 = (correlation1 + correlation2 + correlation3) / 3
            result[CommonVars.KendallTau] = correlation1
            result[CommonVars.Spearman] = correlation2
            result[CommonVars.Pearson] = correlation3
            result[CommonVars.AvgCorrelation] = correlation4
        else:
            raise NotImplementedError(measure_metrics + " is not implemented")

        return result

    @staticmethod
    def compare(ytest, test_pred):
        ytest = np.array(ytest)
        test_pred = np.array(test_pred)
        METRICS = [
            "mae",
            "rmse",
            "pearson",
            "spearman",
            "kendalltau",
            "kt_2dec",
            "kt_1dec",
            "precision_10",
            "precision_20",
            "full_ytest",
            "full_testpred",
        ]
        metrics_dict = {}

        try:
            metrics_dict["mae"] = np.mean(abs(test_pred - ytest))
            metrics_dict["rmse"] = metrics.mean_squared_error(
                ytest, test_pred, squared=False
            )
            metrics_dict["pearson"] = np.abs(np.corrcoef(ytest, test_pred)[1, 0])
            metrics_dict["spearman"] = stats.spearmanr(ytest, test_pred)[0]
            metrics_dict["kendalltau"] = stats.kendalltau(ytest, test_pred)[0]
            metrics_dict["kt_2dec"] = stats.kendalltau(
                ytest, np.round(test_pred, decimals=2)
            )[0]
            metrics_dict["kt_1dec"] = stats.kendalltau(
                ytest, np.round(test_pred, decimals=1)
            )[0]
            print("ytest = ", ytest)
            print("test_pred = ", test_pred)
            for k in [10, 20]:
                top_ytest = np.array(
                    [y > sorted(ytest)[max(-len(ytest), -k - 1)] for y in ytest]
                )
                top_test_pred = np.array(
                    [
                        y > sorted(test_pred)[max(-len(test_pred), -k - 1)]
                        for y in test_pred
                    ]
                )
                metrics_dict["precision_{}".format(k)] = (
                        sum(top_ytest & top_test_pred) / k
                )
            metrics_dict["full_ytest"] = ytest.tolist()
            metrics_dict["full_testpred"] = test_pred.tolist()

        except:
            for metric in METRICS:
                metrics_dict[metric] = float("nan")
        if np.isnan(metrics_dict["pearson"]) or not np.isfinite(
                metrics_dict["pearson"]
        ):
            logger.info("Error when computing metrics. ytest and test_pred are:")
            logger.info(ytest)
            logger.info(test_pred)

        return metrics_dict
