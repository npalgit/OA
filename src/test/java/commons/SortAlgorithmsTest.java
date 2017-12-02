package commons;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by tao on 9/10/17.
 */
public class SortAlgorithmsTest {

    SortAlgorithms sa=null;

    @Before
    public void setup(){
        sa=new SortAlgorithms();
    }
    @Test
    public void countSort() throws Exception {
        int[]nums={5,-5,3,2,1};
        sa.countSort(nums);
        for(int x:nums)
            System.out.println(x);
    }

}