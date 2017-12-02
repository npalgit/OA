import commons.DoubleListNode;
import commons.TreeNode;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by tao on 9/4/17.
 */
public class WePayTest {

    WePay wePay = null;

    public void printList(DoubleListNode head){
        while(head!=null){
            System.out.println(head.val);
            head=head.next;
        }
    }

    @Before
    public void setup(){
        wePay=new WePay();
    }
    @Test
    public void maxSubArray() throws Exception {
        int []nums={-2,1,-3,4,-1,2,1,-5,4};
        System.out.println(wePay.maxSubArray(nums));
    }

    @Test
    public void testGetPath()throws Exception{
//        TreeNode node =new TreeNode(1);
//        node.left=new TreeNode(2);
//        node.right=new TreeNode(3);
//        node.left.left=new TreeNode(4);
//        node.left.right=new TreeNode(5);
//        node.right.left=new TreeNode(6);
//        node.right.right=new TreeNode(7);
//        node.right.right.right=new TreeNode(9);
//        printList(wePay.getPath(node,node.right.right.right));
        TreeNode node =new TreeNode(8);
        node.left=new TreeNode(3);
        node.right=new TreeNode(10);
        node.right.right=new TreeNode(14);
        node.right.right.left=new TreeNode(13);
        node.right.right.left.left=new TreeNode(100);
        node.right.right.right=new TreeNode(15);
        node.left.right=new TreeNode(6);
        node.left.right.right=new TreeNode(7);
        node.left.left=new TreeNode(1);
        node.left.right.left=new TreeNode(4);
        wePay.printDiagnol(node);
    }

    @Test
    public void testThreeSum()throws Exception{
        int[]nums={1,2,3,4,5,6,7,8,9,10,11,12,15};
        System.out.println(wePay.getThreeSum(nums));
        System.out.println(wePay.getThreeSumBetter(nums));
        System.out.println(wePay.subCount(nums,3));
    }

}