package design;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by tao on 9/21/17.
 */
public class AllOneTest {

    @Test
    public void testAllOne(){
        AllOne allOne = new AllOne();
        allOne.inc("a");
        allOne.inc("b");
        allOne.inc("b");
        allOne.inc("b");
        allOne.inc("b");
        allOne.dec("b");
        allOne.dec("b");
        System.out.println(allOne.getMaxKey());
        System.out.println(allOne.getMinKey());
    }

}