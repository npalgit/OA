package medium;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by tao on 9/14/17.
 */
public class PureStorageTest {


    PureStorage ps = null;

    @Before
    public void setup(){
        ps = new PureStorage();
    }
    @Test
    public void longestPalindrome() throws Exception {
        System.out.println(ps.longestPalindrome(""));
    }

    @Test
    public void testSearch()throws Exception{
        int []s={1,2,3,4,5};
        System.out.println(ps.sorted_search(s,3));
    }

}