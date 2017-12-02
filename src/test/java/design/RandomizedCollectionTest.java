package design;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by tao on 9/14/17.
 */
public class RandomizedCollectionTest {

    @Test
    public void testRandom()throws Exception{
        RandomizedCollection rc = new RandomizedCollection();
        rc.insert(10);
        rc.insert(10);
        rc.insert(20);
        rc.insert(20);
        rc.insert(30);
        rc.insert(30);
        rc.remove(10);
        rc.remove(10);
        rc.remove(30);
        rc.remove(30);
    }

}