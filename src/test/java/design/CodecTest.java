package design;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import static org.junit.Assert.*;

/**
 * Created by tao on 9/3/17.
 */
public class CodecTest {


    @Test
    public void testCodec(){
        Codec codec = new Codec();
        Map<Integer,Integer>map = new HashMap<>();
        List<String> strs=new ArrayList<>();
        strs.add("123@123");
        strs.add("world");
        strs.add("hello");
        strs.add("@tao@wang");
        System.out.println(codec.decode(codec.encode(strs)));
    }

}