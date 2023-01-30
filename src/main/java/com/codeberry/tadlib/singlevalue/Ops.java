package com.codeberry.tadlib.singlevalue;

import static com.codeberry.tadlib.singlevalue.GradLink.dependency;
import static java.util.Arrays.asList;
import static java.util.Collections.singletonList;

public class Ops {

    public static Value add(Value a, Value b) {
        GradFunc gradFn = grad -> grad;
        return new Value(a.v + b.v,
            asList(dependency(a, gradFn), dependency(b, gradFn)));
    }

    public static Value sub(Value a, Value b) {
        return new Value(a.v - b.v, asList(
                dependency(a, g -> +g),
                dependency(b, g -> -g)));
    }

    public static Value mul(Value a, Value b) {
        return new Value(a.v * b.v, asList(
                dependency(a, g -> g * b.v),
                dependency(b, g -> g * a.v)));
    }

    public static Value sqr(Value a) {
        return new Value(a.v * a.v,
            singletonList(dependency(a, grad -> 2 * a.v * grad)));
    }

    public static Value sin(Value a) {
        return new Value(Math.sin(a.v), singletonList(dependency(a, grad -> Math.cos(a.v) * grad)));
    }

    public static Value neg(Value a) {
        return new Value(-a.v, singletonList(dependency(a, grad -> -grad)));
    }

    public static Value tanh(Value a) {
        double v = Math.tanh(a.v);
        return new Value(v, singletonList(dependency(a, grad -> (1 - v * v) * grad)));
    }

    public static Value pow(Value a, Value b) {
        double v = Math.pow(a.v, b.v);
        return new Value(v, asList(
            dependency(a, g -> b.v * Math.pow(a.v, b.v - 1) * g),
            dependency(b, g -> v * Math.log(a.v) * g)));
    }
}
