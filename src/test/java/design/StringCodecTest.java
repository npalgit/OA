package design;

import commons.TreeNode;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by tao on 9/3/17.
 */


public class StringCodecTest {

    @Test
    public void testDecode(){
        StringCodec codec = new StringCodec();
        TreeNode node = new TreeNode(1);
        node.left=new TreeNode(2);
        node.right=new TreeNode(3);
        TreeNode node1=codec.deserialize(codec.serialize(node));
        System.out.println(node1.val);
    }

}