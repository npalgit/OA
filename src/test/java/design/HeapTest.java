package design;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by tao on 10/18/17.
 */
public class HeapTest {

    Heap heap = null;

    @Test
    public void testHeap(){
        int []nums={10,2,3,4,5};
        heap = new Heap(nums);
        //heap.buildHeap(nums.length);
        int []ans = heap.sort();
        for(int x:ans)
            System.out.println(x);
    }

}