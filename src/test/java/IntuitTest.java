import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by tao on 10/18/17.
 */
public class IntuitTest {

    Intuit intuit = null;

    @Before
    public void setup(){
        intuit = new Intuit();
    }

    @Test
    public void tasksByLevel() throws Exception {
        List<List<String>>inputs = new ArrayList<>();
        List<String>input1 = new ArrayList<>(Arrays.asList("cook","eat"));
        List<String>input2 = new ArrayList<>(Arrays.asList("study","eat"));
        List<String>input3 = new ArrayList<>(Arrays.asList("sleep","study"));
        inputs.add(input1);
        inputs.add(input2);
        inputs.add(input3);
        System.out.println(intuit.tasksByLevel(inputs));
    }

    @Test
    public void testLongestHistory(){
        String []user1= {"3234.html", "xys.html", "7hsaa.html"};
        String []user2 = {"3234.html", "sdhsfjdsh.html", "xys.html", "7hsaa.html"};
        System.out.println(intuit.longestCommonHistory(user1,user2));
    }

    @Test
    public void testNolessThanK(){
        int []nums={3, 6, 2, 8, 7, 9};
        System.out.println(intuit.NoLessThanK(nums,10));
    }

    @Test
    public void countBy(){
        //System.out.println(intuit.countBinarySubstrings("00110"));
    }

}