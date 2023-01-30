package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.array.TArrayFactory;
import com.codeberry.tadlib.array.util.DimensionUtils;
import com.codeberry.tadlib.nn.loss.SoftmaxCrossEntropyLoss;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.Shape;
import com.codeberry.tadlib.util.MultiThreadingSupport;

import java.util.*;

import static com.codeberry.tadlib.array.TArrayFactory.fillLike;
import static com.codeberry.tadlib.array.TArrayFactory.ones;
import static com.codeberry.tadlib.array.util.DimensionUtils.validateBroadcastShapes;
import static com.codeberry.tadlib.array.util.DimensionUtils.validateMatMulShapes;
import static com.codeberry.tadlib.provider.ProviderStore.shape;
import static com.codeberry.tadlib.provider.java.NDArray.DimKeepRemove.KEEP_DIM;
import static com.codeberry.tadlib.provider.java.NDArray.DimKeepRemove.REMOVE_DIM;
import static com.codeberry.tadlib.provider.java.NDArray.*;
import static com.codeberry.tadlib.tensor.GradLink.grad;
import static com.codeberry.tadlib.util.MultiThreadingSupport.TaskRange.taskRange;
import static java.lang.Boolean.TRUE;
import static java.util.Arrays.asList;
import static java.util.Arrays.stream;
import static java.util.Collections.singletonList;
import static java.util.Collections.synchronizedMap;
import static java.util.stream.Collectors.toList;

public abstract class Ops {
    public static final double EPSILON = Double.MIN_NORMAL; //0.000001;

    public static Tensor MATMUL(Tensor a, Tensor b) {
        DimensionUtils.MatMulParams params = DimensionUtils.MatMulParams.expandSingleDimArrays(a.shape(), b.shape(), Shape::shape);
        Shape LEFT_SHAPE = params.leftShape, RIGHT_SHAPE = params.rightShape;

        validateMatMulShapes(LEFT_SHAPE, RIGHT_SHAPE);
        validateBroadcastShapes(LEFT_SHAPE, RIGHT_SHAPE, -3);

        Shape outShape = evalMatMulShape(LEFT_SHAPE, RIGHT_SHAPE);
        int[] outShapeIndexArray = outShape.newIndexArray();
        double[] data = new double[outShape.size];

        int outputRows = outShape.at(-2);

        GradFunc gF_a = grad ->
                aggregateBroadcastedDims(a, grad.matmul(b.val().transposeLast2D()));
        GradFunc gF_b = grad ->
                aggregateBroadcastedDims(b, a.val().transposeLast2D().matmul(grad));

        MultiThreadingSupport.TaskRange totalRange = taskRange(0, outputRows).withMinimumWorkLength(decideMinimumRowsPerThread(LEFT_SHAPE, outShape));

        return new Tensor(v -> {
            NDArray left = matMulLR(params.promoteLeft, a, LEFT_SHAPE);
            NDArray right = matMulLR(params.promoteRight, b, RIGHT_SHAPE);
            v.set(MultiThreadingSupport.<double[]>multiThreadingSupportRun(
                    totalRange,
                    range -> NDArray.matmul(range, left, right, data, outShape, outShapeIndexArray, 0),
                    (_left, ignored_) -> _left)
            );
        }, params.revertDimExpandOfOutputShape(outShape), asList(grad(a, gF_a), grad(b, gF_b)));
    }

    private static NDArray matMulLR(boolean params, Tensor a, Shape LEFT_SHAPE) {
        return params ? new NDArray(a.val().data, LEFT_SHAPE) : a.val();
    }

    public static Tensor matmul(Tensor a, Tensor b) {
        NDArray y = a.val().matmul(b.val());
        return new Tensor(y, asList(
                grad(a, g -> aggregateBroadcastedDims(a, g.matmul(b.val().transposeLast2D()))),
                grad(b, g -> aggregateBroadcastedDims(b, a.val().transposeLast2D().matmul(g)))));
    }

    public static Tensor NEGATE(Tensor x) {
        return new Tensor(y -> {
            y.set(x.val().mul(-1));
        }, x.shape(), singletonList(grad(x, NDArray::negate)));
    }

    public static Tensor negate(Tensor a) {
        return new Tensor(a.val().negate(), singletonList(grad(a, NDArray::negate)));
    }

    public static Tensor sub(Tensor a, Tensor b) {
        return add(a, negate(b));
    }

    public static Tensor SUB(Tensor a, Tensor b) {
        return ADD(a, NEGATE(b));
    }

    public static Tensor add(Tensor... tensors) {
        NDArray y = stream(tensors)
                .map(Tensor::val)
                .reduce(NDArray::add)
                .orElseGet(TArrayFactory::zerosShaped);

        return new Tensor(y, stream(tensors)
                .map(t -> grad(t, funcGradientAdd(t)))
                .collect(toList()));
    }

    public static Tensor add(Tensor a, Tensor b) {
        return new Tensor(a.val().add(b.val()), asList(grad(a, funcGradientAdd(a)), grad(b, funcGradientAdd(b))));
    }

    public static Tensor ADD(Tensor a, Tensor b) {
        return new Tensor(v -> v.set(a.val().add(b.val())), a.shape(), asList(grad(a, funcGradientAdd(a)), grad(b, funcGradientAdd(b))));
    }

    public static Tensor DENSE(Tensor x, int outputSize) {
        return DENSE(x, outputSize, true);
    }

    public static Tensor DENSE(Tensor x, int outputSize, boolean bias) {
        var xShape = x.shape();
        var wShape = shape(xShape.at(-1), outputSize);
        Tensor W = new Tensor(wShape).optimizable();
        Tensor B = bias ? new Tensor(shape(outputSize)).optimizable() : null;
        Tensor y = MATMUL(/*FLATTEN*/(x), W);
        return bias ? ADD(y, B) : y;
    }

    public static Tensor add(Tensor a, double constant) {
        return new Tensor(a.val().add(constant), singletonList(grad(a, funcGradientAdd(a))));
    }

    public static Tensor ADD(Tensor x, double constant) {
        return new Tensor(y -> y.set(x.val().add(constant)), x.shape(), singletonList(grad(x, funcGradientAdd(x))));
    }

    public static Tensor MUL(Tensor x, double constant) {
        return new Tensor(y -> y.set(x.val().mul(constant)), x.shape(), singletonList(grad(x, funcGradientMul(x, constant))));
    }

    public static Tensor SUB(Tensor x, double constant) {
        return ADD(x, -constant);
    }

    private static GradFunc funcGradientAdd(Tensor tensor) {
        return grad -> aggregateBroadcastedDims(tensor, grad);
    }

    public static Tensor sqr(Tensor a) {
        return new Tensor(a.val().sqr(), singletonList(grad(a,
                grad -> grad.mul(a.val()).mul(2))));
    }

    public static Tensor sqrt(Tensor a) {
        return new Tensor(a.val().sqrt(), singletonList(grad(a, grad ->
                grad.mul(a.val().pow(-0.5).mul(0.5)))));
    }

    public static Tensor softmax(Tensor input) {
        NDArray softmax = input.val().softmax();

        // Main source: https://aerinykim.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
        GradFunc gF = grad -> {
            Shape backGradShape = grad.shape;

            NDArray smByColumn = softmax.reshape(softmax.shape.appendDim(1));
            NDArray smByRow = smByColumn.transposeLast2D();
            NDArray selfMatMuled = smByColumn.matmul(smByRow);
            NDArray smDiagonal = softmax.diag();
            NDArray separatedGrad = smDiagonal.sub(selfMatMuled);

            NDArray gradByColumn = grad.reshape(backGradShape.appendDim(1));
            NDArray gradWithExtraDim = separatedGrad.matmul(gradByColumn);

            return gradWithExtraDim.reshape(backGradShape);
        };

        return new Tensor(softmax, singletonList(grad(input, gF)));
    }

    public static Tensor div(Tensor a, Tensor b) {
        NDArray y = a.val().div(b.val());

        GradFunc gF_a = grad -> aggregateBroadcastedDims(a, grad).div(b.val());

        //  calced_divisor_grad = tf.reduce_sum(fake_grad * (-dividend / (divisor**2)), axis=(0,1))
        GradFunc gF_b = grad -> aggregateBroadcastedDims(b,
                grad.mul(a.val().negate()).div(b.val().sqr()));

        return new Tensor(y, asList(grad(a, gF_a), grad(b, gF_b)));
    }

    public static Tensor mul(Tensor a, Tensor b) {
        return new Tensor(a.val().mul(b.val()), asList(
                grad(a, funcGradientMul(a, b)),
                grad(b, funcGradientMul(b, a))));
    }

    public static Tensor SQR(Tensor x) {
        return MUL(x, x);
    }

    public static Tensor MUL(Tensor a, Tensor b) {
        validateBroadcastShapes(a.shape(), b.shape(), -1);

        return new Tensor(v -> {
            v.set(a.val().mul(b.val())); //TODO direct write
        }, evalBroadcastOutputShape(a.shape(), b.shape()), asList(
                grad(a, funcGradientMul(a, b)),
                grad(b, funcGradientMul(b, a))));
    }

    private static GradFunc funcGradientMul(Tensor self, Tensor other) {
        return grad -> aggregateBroadcastedDims(self, grad.mul(other.val()));
    }

    private static GradFunc funcGradientMul(Tensor self, double other) {
        return grad -> aggregateBroadcastedDims(self, grad.mul(other));
    }

    private static NDArray aggregateBroadcastedDims(Tensor self, NDArray grad) {

        NDArray gr = grad;
        NDArray ndArray2 = self.val();
        int missingDims = gr.shape.dimCount - ndArray2.shape.dimCount;
        if (missingDims >= 1)
            gr = gr.sumFirstDims(missingDims, REMOVE_DIM);

        NDArray ndArray1 = self.val();
        if (ndArray1.shape.hasSingleDims())
            gr = gr.sum(self.val().shape.forEachDim(dim -> dim == 1, boolean[]::new), KEEP_DIM);

        return gr;
    }

    public static Tensor sum(Tensor a) {
        return new Tensor(a.val().sum(), singletonList(
            grad(a, grad -> fillLike(a.val().shape, grad))));
    }

    public static Tensor SUM(Tensor a) {
        return new Tensor(y -> y.set(a.val().sum()), Shape.zeroDim, singletonList(grad(a, grad -> fillLike(a.shape(), grad))));
    }

    public static Tensor conv2d(Tensor input, Tensor filter) {
        NDArray y = input.val().conv2d(filter.val());

        GradFunc gF_Input = grad -> {
            NDArray transposed = filter.val().transpose(0, 1, 3, 2);
            NDArray fT = transposed.normalOrderedCopy();
            NDArray fRotated = fT.rot180(0, 1);

            NDArray ndArray1 = filter.val();
            int offsetY = (ndArray1.shape.at(0) % 2 == 0) ? -1 : 0;
            NDArray ndArray = filter.val();
            int offsetX = (ndArray.shape.at(1) % 2 == 0) ? -1 : 0;
            return grad.conv2d(fRotated, offsetY, offsetX);
        };
        GradFunc gF_Filter = grad -> grad.calcConv2dFilterGradient(input.val(), filter.val());

        return new Tensor(y,
                asList(grad(input, gF_Input),
                        grad(filter, gF_Filter)));
    }

    public static Tensor maxpool2d(Tensor input, int size) {
        return new Tensor(((MaxPool2dResult) input.val().maxPool2d(size)).output(), singletonList(grad(input, grad -> grad.maxPool2dGrad(input.val().maxPool2d(size)))));
    }

    public static Tensor flatten(Tensor input) {
        Shape inputShape = input.val().shape;
        return new Tensor(input.val().reshape(inputShape.at(0),
                calcFlattenExampleSize(inputShape)),
                singletonList(grad(input, grad -> grad.reshape(inputShape))));
    }

    public static Tensor FLATTEN(Tensor input) {
        Shape inputShape = input.shape();
        return new Tensor(v -> v.set(input.val()),
                inputShape.reshape(inputShape.at(0), calcFlattenExampleSize(inputShape)),
                singletonList(grad(input, grad -> grad.reshape(inputShape))));
    }

    public static int calcFlattenExampleSize(Shape inputShape) {
        int size = 1;
        for (int i = 1; i < inputShape.dimCount; i++)
            size *= inputShape.at(i);
        return size;
    }

    public static Tensor reshape(Tensor input, int... dims) {
        NDArray ndArray = input.val();
        return new Tensor(ndArray.reshape(dims),
            singletonList(grad(input, grad -> grad.reshape(ndArray.shape))));
    }

    public static Tensor relu(Tensor input) {
        NDArray.ReluResult result = input.val().relu(0.0);
        return new Tensor(result.getOutput(), singletonList(grad(input, grad -> grad.mul(result.createMask()))));
    }

    public static Tensor RELU(Tensor x) {
        ReluResult[] xr = new ReluResult[1];
        return new Tensor(y -> {
            xr[0] = x.val().relu(0);
            y.set(xr[0].getOutput());
        }, x.shape(), singletonList(grad(x, g -> g.mul(xr[0].createMask()))));
    }

    public static Tensor leakyRelu(Tensor input, double leakyScale) {
        NDArray.ReluResult result = input.val().relu(leakyScale);
        return new Tensor(result.getOutput(), singletonList(grad(input, grad -> grad.mul(result.createMask()))));
    }

    public static Tensor sumSoftmaxCrossEntropy(Tensor labelsOneHot, Tensor prediction) {
        NDArray softmax = prediction.val().softmax();

        NDArray oneHotArray = labelsOneHot.val();
        NDArray cost = SoftmaxCrossEntropyLoss.sumSoftmaxCrossEntropy(softmax, oneHotArray);

        GradFunc gF = grad -> grad.softMaxCrossEntropyGrad(softmax, oneHotArray);

        return new Tensor(cost, singletonList(grad(prediction, gF)));
    }

    public static Tensor dropout(Tensor input, Random rnd, double dropoutKeep, RunMode runMode) {
        if (runMode == RunMode.TRAINING) {
            NDArray.DropOutResult result = input.val().dropOut(rnd, dropoutKeep);
            NDArray output = result.output();

            GradFunc gF = grad -> grad.mul(result.createMask());

            return new Tensor(output, singletonList(grad(input, gF)));
        }
        return input;
    }

    public static Tensor mean(Tensor input, int... axis) {
        NDArray ndArray = input.val();
        Shape inputShape = ndArray.shape;
        boolean[] dimsToCollapse = inputShape.newCollapseArray();
        for (int axi : axis) {
            dimsToCollapse[axi] = TRUE;
        }
        int elementsSummed = 1;
        for (int i = 0; i < inputShape.dimCount; i++) {
            if (dimsToCollapse[i]) {
                elementsSummed *= inputShape.at(i);
            }
        }

        NDArray sum = input.val().sum(dimsToCollapse, REMOVE_DIM);
        NDArray mean = sum.div(elementsSummed);

        int[] broadcastDims = inputShape.toDimArray();
        for (int d = 0; d < dimsToCollapse.length; d++) {
            if (dimsToCollapse[d])
                broadcastDims[d] = 1;
        }

        int ELEMENTS_SUMMED = elementsSummed;
        GradFunc gF = grad -> {
            NDArray broadcastCompatGrad = grad.reshape(broadcastDims);
            NDArray onesInInputShape = ones(inputShape);
            NDArray gradInInputShape = onesInInputShape.mul(broadcastCompatGrad);

            return gradInInputShape.div(ELEMENTS_SUMMED);
        };

        return new Tensor(mean, singletonList(grad(input, gF)));
    }

    public static Tensor sum(Tensor input, int... axis) {
        NDArray ndArray = input.val();
        Shape inputShape = ndArray.shape;
        boolean[] dimsToCollapse = inputShape.newCollapseArray();
        for (int axi : axis)
            dimsToCollapse[axi] = TRUE;

        int[] broadcastDims = inputShape.toDimArray();
        for (int d = 0; d < dimsToCollapse.length; d++) {
            if (dimsToCollapse[d]) {
                broadcastDims[d] = 1;
            }
        }

        GradFunc gF = grad -> ones(inputShape).mul(grad.reshape(broadcastDims));

        NDArray sum = input.val().sum(dimsToCollapse, REMOVE_DIM);
        return new Tensor(sum, singletonList(grad(input, gF)));
    }

    public static Tensor SUM(Tensor x, int... axis) {
        Shape inputShape = x.shape();
        boolean[] dimsToCollapse;

        dimsToCollapse = inputShape.newCollapseArray();
        for (int axi : axis)
            dimsToCollapse[axi] = true;

        int[] broadcastDims = inputShape.toDimArray();
        for (int d = 0; d < dimsToCollapse.length; d++)
            if (dimsToCollapse[d])
                broadcastDims[d] = 1;

        return new Tensor(y -> {
            y.zero();
            y.sum(x.val(), dimsToCollapse, REMOVE_DIM);
        }, toPhysicalShape(inputShape, dimsToCollapse), singletonList(grad(x,
                grad -> ones(inputShape).mul(grad.reshape(broadcastDims)))));
    }

    public static Tensor log(Tensor x) {

        return new Tensor(x.val().log(), singletonList(grad(x, grad -> grad.div(x.val()))));
    }

    public static Tensor concat(int axis, Tensor... tensors) {
        NDArray first = tensors[0].val();
        NDArray[] rest = new NDArray[tensors.length - 1];
        for (int i = 1; i < tensors.length; i++) {
            rest[i - 1] = tensors[i].val();
        }
        NDArray concat = first.concat(rest, axis);

        int[] axisLens = Arrays.stream(tensors)
                .map(Tensor::val)
                .map(ndArray -> ndArray.shape)
                .mapToInt(shape -> shape.at(axis))
                .toArray();
        GradSplitter gradSplitter = new GradSplitter(axis, axisLens);

        List<GradLink> links = new ArrayList<>(tensors.length);
        for (int i = 0; i < tensors.length; i++)
            links.add(grad(tensors[i], gradSplitter.getGradOfPart(i)));

        return new Tensor(concat, links);
    }

    public enum RunMode {
        TRAINING, INFERENCE
    }

    private static class GradSplitter {
        private final int axis;
        private final int[] axisLens;

        private final Map<NDArray, List<NDArray>> splitGradientsByMainGrad = synchronizedMap(new IdentityHashMap<>());

        GradSplitter(int axis, int[] axisLens) {
            this.axis = axis;
            this.axisLens = axisLens;
        }

        GradFunc getGradOfPart(int partIndex) {
            return gradient -> {
                List<NDArray> splitGrads = ensureSplitResult(gradient);
                return splitGrads.get(partIndex);
            };
        }

        @SuppressWarnings("SynchronizationOnLocalVariableOrMethodParameter")
        private List<NDArray> ensureSplitResult(NDArray grad) {
            synchronized (grad) {
                return splitGradientsByMainGrad.computeIfAbsent(grad, gr -> gr.split(axis, axisLens));
            }
        }
    }
}
