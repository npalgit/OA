import commons.TreeNode;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class ForUSAllTest {

    ForUSAll fu = null;

    @Before
    public void setup(){
        fu = new ForUSAll();
    }
    @Test
    public void solution() throws Exception {

        TreeNode node = new TreeNode(1);
        node.left = new TreeNode(2);
        node.right = new TreeNode(3);
        node.left.right = new TreeNode(4);
        node.right.right = new TreeNode(6);
        node.right.left = new TreeNode(5);
        node.right.left.left = new TreeNode(7);
        node.right.left.right = new TreeNode(8);
        node.right.right.left = new TreeNode(9);
        node.right.right.right = new TreeNode(10);
        node.right.right.right.left = new TreeNode(11);
        System.out.println();
        System.out.println(fu.largestCompleteTree(node));
    }

    @Test
    public void test(){
        int []movies = {6,1,4,6,3,2,7,4};
        System.out.println(fu.solution(movies,3,2));
    }

}