package com.codeberry.tadlib.nn.layer;

import com.codeberry.tadlib.nn.Model.IterationInfo;
import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.Shape;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Random;

import static com.codeberry.tadlib.nn.layer.Layer.ForwardResult.result;
import static com.codeberry.tadlib.tensor.Ops.RunMode;
import static com.codeberry.tadlib.tensor.Ops.add;
import static java.lang.Math.toIntExact;

public class ProportionalDropOutLayer implements Layer {
    private final double strength;
    private final Shape outputShape;

    public ProportionalDropOutLayer(Shape inputShape, Builder params) {
        this.strength = params.strength;

        outputShape = inputShape;
    }

    @Override
    public ForwardResult forward(Random rnd, Tensor inputs, RunMode runMode, IterationInfo iterationInfo) {
        return result(getForwardResult(rnd, inputs, runMode));
    }

    private Tensor getForwardResult(Random rnd, Tensor inputs, RunMode runMode) {
        if (runMode == RunMode.TRAINING) {

            Shape shape = inputs.shape();
            double[] rndVals = new double[toIntExact(shape.size)];
            for (int i = 0; i < rndVals.length; i++) {
                rndVals[i] = rnd.nextGaussian() * strength;
            }

            NDArray randomArr = ProviderStore.array(rndVals, shape);
            Tensor randomTensor = Tensor.constant(randomArr.mul(inputs.val()));

            return add(inputs, randomTensor);
        }
        return inputs;
    }

    @Override
    public Shape getOutputShape() {
        return outputShape;
    }

    public static class Builder implements LayerBuilder {

        @Override
        public Layer build(Random rnd, Shape inputShape) {
            return new ProportionalDropOutLayer(inputShape, this);
        }

        double strength;

        public static Builder proportionalDropout() {
            return new Builder();
        }

        public Builder strength(double strength) {
            this.strength = strength;
            return this;
        }
    }
}
