package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.util.MatrixTestUtils;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.array.TArrayFactory.range;
import static java.util.Arrays.deepToString;

public class TensorFlattenTest {

    @Test
    public void flatten() {
        NDArray raw = range(3 * 4 * 4);
        Tensor input = new Tensor(raw.reshape(3, 4, 4, 1));

        Tensor flattened = Ops.flatten(input);
        System.out.println(deepToString((Object[]) flattened.val().toDoubles()));

        NDArray g = ProviderStore.array(new double[]{
                100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 1100, 1101, 1102, 1103, 1104, 1105,
                10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 110, 111, 112, 113, 114, 115,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        }).reshape(3, 4 * 4);
        flattened.backward(g);

        NDArray expected = raw.reshape(3, 4 * 4);
        MatrixTestUtils.assertEqualsMatrix(expected.toDoubles(),
                flattened.val().toDoubles());

        System.out.println(deepToString((Object[]) input.grad().toDoubles()));
        MatrixTestUtils.assertEqualsMatrix(g.reshape(3, 4, 4, 1).toDoubles(),
                input.grad().toDoubles());


    }
}
