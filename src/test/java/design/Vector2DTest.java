package design;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by tao on 9/1/17.
 */
public class Vector2DTest {


    @Test
    public void testVector(){
        List<List<Integer>>res=new ArrayList<>();
        List<Integer>res1 = new ArrayList<>(Arrays.asList(1,2));
        List<Integer>res2 = new ArrayList<>(Arrays.asList(3));
        List<Integer>res3 = new ArrayList<>(Arrays.asList(4,5,6));
        List<Integer>res4 = new ArrayList<>();
        res.add(res1);
        res.add(res2);
        res.add(res4);
        res.add(res3);
        Vector2D vec = new Vector2D(res);
        while(vec.hasNext()){
            System.out.println(vec.next());
        }
    }

}