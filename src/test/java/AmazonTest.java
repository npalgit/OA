import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by tao on 10/3/17.
 */
public class AmazonTest {

    Amazon amazon = null;

    @Before
    public void setup(){
        amazon =new Amazon();
    }

    @Test
    public void getSum() throws Exception {
        int []nums={1,2,3,4,5};
        int []res = amazon.getSum(nums,3);
        for(int x:res)
            System.out.println(x);
    }
    @Test
    public void getSumArrayList(){
        List<Integer> nums=new ArrayList<>(Arrays.asList(1,2,3,4,5));

        System.out.println(amazon.GetSum(nums,3));
    }


}