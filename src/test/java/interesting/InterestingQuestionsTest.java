package interesting;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by tao on 9/16/17.
 */
public class InterestingQuestionsTest {


    InterestingQuestions iq =null;

    @Before
    public void setup(){
        iq = new InterestingQuestions();
    }
    @Test
    public void maxSumNoLargerThanK() throws Exception {
        int []nums={1,12,3,14,10,13,15,14,17};
        System.out.println(iq.maxSumNoLargerThanK(nums,17));
        System.out.println(iq.maxSum(nums,17));

    }

    @Test
    public void testMaxi()throws Exception{
        int[]nums1= {2,5,6,4,4,0};
        int []nums2 = {7,3,8,0,6,5,7,6,2};
        int []res = iq.getMaxi(nums1,nums2);
        for(int x:res)
            System.out.print(x+" ");
    }

    @Test
    public void testElimination(){
        System.out.println(iq.lastRemaining(2147483647));
    }

    @Test
    public void testCanWin(){
        System.out.println(iq.canIWin(20,200));
    }

}