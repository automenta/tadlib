package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.nn.Model;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.ValueUpdate;
import com.codeberry.tadlib.util.TrainingData;

import java.util.List;
import java.util.Random;

import static java.util.Collections.singletonList;

class ModelParamEstimator {
    private static final double DELTA = 0.000008;

    private final int rndSeed;
    private final Model model;
    private final int paramIndex;
    private final TrainingData trainingData;

    public ModelParamEstimator(int rndSeed, Model model, int paramIndex, TrainingData trainingData) {
        this.rndSeed = rndSeed;
        this.model = model;
        this.paramIndex = paramIndex;
        this.trainingData = trainingData;
    }

    public double[] estimate() {
        List<Tensor> params = model.getParams();
        Tensor param = params.get(paramIndex);
        int vol = param.shape().size;
        double[] grad = new double[vol];
        NDArray orgArray = param.val();

        int segLen = vol / 4 + 1;
        System.out.println("Param Index: " + paramIndex + " (" + vol + " values)");
        for (int i = 0; i < vol; i++) {
            if (i % segLen == 0) {
                System.out.println("  " + (i * 100 / grad.length) + "%");
            }

            double org = param.val().dataAtIndex(i);

            grad[i] = estimateWithRespectToElement(param, DELTA, i, org);

            param.update((vals, gradient) -> orgArray);
        }

        return grad;
    }

    private double estimateWithRespectToElement(Tensor param, double d, int elementIndex, double orgValue) {
        double valBefore = orgValue - d;
        param.update((vals, gradient) -> vals.withUpdates(singletonList(new ValueUpdate(elementIndex, valBefore))));
        double before = calcCost();

        double valAfter = orgValue + d;
        param.update((vals, gradient) -> vals.withUpdates(singletonList(new ValueUpdate(elementIndex, valAfter))));
        double after = calcCost();

        return (after - before) / (2 * d);
    }

    private double calcCost() {
        Model.PredictionAndLosses predictionAndLosses = model.calcCost(new Random(rndSeed), trainingData.getTrainingBatchAll(), new Model.IterationInfo(0, 0, 1));
        return (double) predictionAndLosses.totalLoss.toDoubles();
    }
}
