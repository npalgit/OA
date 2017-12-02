import commons.ListNode;
import commons.TreeNode;
import org.junit.Before;
import org.junit.Test;

import javax.annotation.Resource;

import static org.junit.Assert.*;

/**
 * Created by tao on 10/14/17.
 */
public class MicrosoftTest {


    Microsoft soft;

    @Before
    public void steup(){
        soft = new Microsoft();
    }

    @Test
    public void replaceN() throws Exception {
        System.out.println(soft.replaceN("hehheh\n\nthjpih\n"));
    }

    @Test
    public void testRemove(){
        System.out.println(soft.removeSpace("     ahhh f hh    hh  hhhff   ff  ff     ff    "));
    }

    @Test
    public void testMinRight(){
        TreeNode node = new TreeNode(5);
        node.left = new TreeNode(3);
        node.right =  new TreeNode(7);
        node.left.left=new TreeNode(2);
        //node.left.right=new TreeNode(4);
        node.right.left=new TreeNode(6);
        node.right.right=new TreeNode(8);
        System.out.println(soft.getMinRightChildren(node));
    }

    @Test
    public void testChange(){
        System.out.println(soft.change("aaabbbccceddaaa"));
    }

    @Test
    public void testSprialOrder(){
        int[][]matrix={{1,2,3},{4,5,6},{7,8,9}};
        soft.spiralOrder(matrix);
    }

    @Test
    public void testBubble(){
        int []nums={2,-4,3,4,6,5};
        //soft.bubbleSort(nums);
//        soft.quickSort(nums);
//        for(int x:nums)
//            System.out.println(x);

//        for(int i=1;i<=nums.length;++i)
//            System.out.println(soft.findKth(nums,i));


        soft.mergeSort(nums);
        for(int x:nums)
            System.out.println(x);
    }

    public void printf(ListNode node){
        while(node!=null){
            System.out.println(node.val);
            node=node.next;
        }
    }
    @Test
    public void testChangeList(){
        ListNode node =new ListNode(1);
        node.next=new ListNode(2);
        node.next.next=new ListNode(3);
        node.next.next.next=new ListNode(4);
        node.next.next.next.next=new ListNode(5);
        node.next.next.next.next.next=new ListNode(6);
        node.next.next.next.next.next.next=new ListNode(7);
        ListNode head=soft.changeListIterative(node);
        printf(head);
    }

    @Test
    public void testBuild(){
        int []nums ={1, 7, 5, 50, 40, 10};
        TreeNode node=soft.build(nums,0,nums.length-1);
        System.out.println(node.val);
    }

    @Test
    public void testMergeSortList(){
        ListNode node = new ListNode(5);
        node.next =  new ListNode(4);
        node.next.next = new ListNode(-4);
        node.next.next.next = new ListNode(-7);
        node.next.next.next.next = new ListNode(3);
        node=soft.sortList(node);
        printf(node);
    }

    @Test
    public void testFrequency(){
        soft.frequencySort("beeeerr");
    }

    @Test
    public void testBinary(){
        System.out.println(soft.getBinary(-1));
        System.out.println(Integer.toBinaryString(10));
        int a =10,b=8;
        System.out.println(a^b);
        System.out.println((~a&b|a&~b));

    }

    @Test
    public void testMatrix(){
        int[][]maxtrix = {{1,2,3},{4,5,6},{7,8,9},{10,11,12}};
        soft.printMatrix(maxtrix);
    }
}