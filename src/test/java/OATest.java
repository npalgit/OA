import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by tao on 8/12/17.
 */
public class OATest {



    OA oa =null;


    @Before
    public void setup(){
        oa=new OA();
    }
    @Test
    public void printPrime() throws Exception {
        oa.printPrime(100);
    }

    @Test
    public void getPower()throws Exception{
        List<Integer> res=oa.getPowerNumber1(10);
        System.out.println(res.get(47686));
        //assertEquals((long)res.get(47686),(long)oa.getPowerNumber(47687));
    }

}