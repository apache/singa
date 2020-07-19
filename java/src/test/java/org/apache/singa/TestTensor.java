/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package test;

import junit.framework.*;
import org.apache.singa.swig.*;

import static org.junit.Assert.*;

public class TestTensor extends TestCase {

    protected void setUp() {
        System.loadLibrary("singa_wrap");
    }

    public void testTensorFunc() {
        Shape s = new Shape(2);
        s.set(0, 2);
        s.set(1, 3);

        Tensor t1 = new Tensor(s);
        t1.SetFloatValue(0.1f);
        Tensor t2 = singa_wrap.Square(t1);
        float[] data = new float[6];

        t2.GetFloatValue(data, 6);
        for (int i = 0; i < 6; i++)
            assertEquals(data[i], 0.01, 1e-4);

        for (int i = 0; i < 6; i++)
            data[i] = i * 1.0f;
        Tensor t3 = new Tensor(s);
        t3.CopyFloatDataFromHostPtr(data, 6);

        t3.GetFloatValue(data, 6);
        for (int i = 0; i < 6; i++)
            assertEquals(data[i], i * 1.0f, 1e-4);
    }
}
