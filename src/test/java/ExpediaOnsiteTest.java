import design.MyException;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by tao on 10/17/17.
 */
public class ExpediaOnsiteTest {

    ExpediaOnsite expediaOnsite = null;

    @Before
    public void setup(){
        expediaOnsite = new ExpediaOnsite();
    }

    @Test
    public void autoComplete(){
        Trie t = new Trie();
        String []words ={ "abc","cde","abccc","abcyui"};
        for(String word:words)
            t.insert(word);
        System.out.println(t.getAllWords("abc"));
    }

    @Test
    public void testCompress(){
        System.out.println(expediaOnsite.compress("aaaabbbccc"));
    }

    @Test
    public void testTriangle(){
        int []nums={2,2,3,4};
        System.out.println(expediaOnsite.numberOfTriangle(nums));
    }

    @Test
    public void testException(){
        if(true)
            throw new MyException("should be terminated");
    }
}