import commons.ListNode;
import commons.TreeNode;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by tao on 9/29/17.
 */
public class ExpediaTest {

    Expedia expedia = null;
    @Before
    public void setup(){
        expedia = new Expedia();
    }

    @Test
    public void sortByFrequence() throws Exception {
        int []nums={3,1,2,2,4};
        int []arr = expedia.sortByFrequence(nums);
        for(int x:arr)
            System.out.println(x);
    }

    @Test
    public void testGetString()throws Exception{
        String[]args1 = {"test","a","v","c","c","v"};
        String []args2 ={"test2","a","c","c"};
        System.out.println(expedia.getString(args1,args2));
    }
    @Test
    public void testGetBits()throws Exception{
        System.out.println(expedia.getBits(77));
    }

    @Test
    public void testInTree(){
        TreeNode node = new TreeNode(5);
        node.left = new TreeNode(3);
        node.right = new TreeNode(7);
        node.left.left=new TreeNode(2);
        node.left.right = new TreeNode(4);
        node.right.left=new TreeNode(6);
        node.right.right=new TreeNode(8);
        System.out.println(expedia.inTree(node,9));
    }

    @Test
    public void testRoman(){
        String[]names = {"PILLIPS II","PILLIPS V","PILLIPS V","PILLIPSSI V"};
        String []sortN = expedia.sortName(names);
        for(String str:sortN)
            System.out.println(str);
    }

    @Test
    public void testMinimumMoves(){
        int []nums={1,2,3};
        System.out.println(expedia.adjustNumber(nums));
    }

    @Test
    public void testRemove(){
        ListNode head = new ListNode(1);
        head.next = new ListNode(2);
        head.next.next = new ListNode(3);
        head.next.next.next = new ListNode(4);
        head.next.next.next.next = new ListNode(5);
        head.next.next.next.next.next = new ListNode(6);
        ListNode node = expedia.removeOdd(head);
        while(node!=null){
            System.out.println(node.val);
            node = node.next;
        }
    }

    @Test
    public void testInside(){
        int x1 =0, y1 =0, x2 =2, y2 =0, x3=1, y3=1, x4=1,y4=2;
        System.out.println(expedia.isInside(x1,y1,x2,y2,x3,y3,x4,y4));
    }

    @Test
    public void testCompression()throws Exception{
        System.out.println(expedia.compression("aabcccb"));
    }

    @Test
    public void testDungeon(){
        int []nums={-1,-2,-3,4,-5,6,4};
        System.out.println(expedia.calculateMinimumHP(nums));
    }

    @Test
    public void testBalanced(){
        int []nums={4,5,6,4,5};
        System.out.println(expedia.balancedSales(nums));
    }
    @Test
    public void testSubCount(){
        int []nums={1,2,3,4,1};
        System.out.println(expedia.subCount(nums,3));
    }

    @Test
    public void testSeparating(){
        int []nums={0,0,0,0,1,1,1,1};
        System.out.println(expedia.separatingStudent(nums));
    }

    @Test
    public void testMissingString(){
        String s = "I am using HackerRank to improve programming";
        String t ="am HackerRank to improve";
        System.out.println(expedia.missingString(s,t));
    }

    @Test
    public void testCountGroup()throws Exception{
        int[][]matrix={{1,1,1,0},{0,0,1,1},{1,1,0,0},{0,1,0,1}};
        System.out.println(expedia.countGroup(matrix));
    }

    @Test
    public void getOddSum()throws Exception{
        int[]nums={3,4,20};
        System.out.println(expedia.getOddDivsiorSum(nums));
    }

    @Test
    public void testMagicString(){
        System.out.println(expedia.magicString(105));
    }

}