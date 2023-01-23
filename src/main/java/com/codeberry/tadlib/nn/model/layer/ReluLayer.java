package com.codeberry.tadlib.nn.model.layer;

import com.codeberry.tadlib.array.Shape;
import com.codeberry.tadlib.nn.model.Model.IterationInfo;
import com.codeberry.tadlib.tensor.Tensor;

import java.util.Random;

import static com.codeberry.tadlib.nn.model.layer.Layer.ForwardResult.result;
import static com.codeberry.tadlib.tensor.Ops.*;

public class ReluLayer implements Layer {
    private final double leakyScale;
    private final Shape outputShape;

    public ReluLayer(Shape inputShape, Builder params) {
        this.leakyScale = params.leakyScale;

        outputShape = inputShape;
    }

    @Override
    public ForwardResult forward(Random rnd, Tensor inputs, RunMode runMode, IterationInfo iterationInfo) {
        return result(leakyRelu(inputs, leakyScale));
    }

    @Override
    public Shape getOutputShape() {
        return outputShape;
    }

    public static class Builder implements LayerBuilder {
        double leakyScale;

        @Override
        public Layer build(Random rnd, Shape inputShape) {
            return new ReluLayer(inputShape, this);
        }

        public static Builder relu() {
            return new Builder();
        }

        public Builder leakyScale(double leakyScale) {
            this.leakyScale = leakyScale;
            return this;
        }

    }
}
