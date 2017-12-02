import commons.Interval;
import commons.TreeNode;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by tao on 10/30/17.
 */
public class FacebookTest {

    Facebook facebook = null;

    @Before
    public void setup(){
        facebook = new Facebook();
    }
    @Test
    public void numDecodingsII() throws Exception {
        System.out.println(facebook.numDecodingsII("**"));
    }

    @Test
    public void testFindAllanagrams(){
        System.out.println(facebook.findAnagrams("abcacba","abc"));
    }

    @Test
    public void testMyPow(){
        int nn = -2147483648;
        System.out.println(-nn);
        System.out.println(facebook.myPowRecursive(1.0,-2147483648));
    }

    @Test
    public void testPreorderIterator(){


        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.right = new TreeNode(7);
        root.left.left = new TreeNode(4);
        root.right.right = new TreeNode(5);
        root.right.left = new TreeNode(100);
        BinaryPostOrderIterator bpi = new BinaryPostOrderIterator(root);
        while(bpi.hasNext()){
            System.out.println(bpi.next());
        }
    }

    @Test
    public void testFindOverlap(){
        Interval []intervals = new Interval[5];
        intervals[0] = new Interval(3,5);
        intervals[1] = new Interval(4,7);
        intervals[2]=  new Interval(0,2);
        intervals[3] = new Interval(10,11);
        intervals[4] = new Interval(7,9);
        //System.out.println(facebook.findOverlap(intervals));
        List<Interval> intervalList = new ArrayList<>(Arrays.asList(intervals));
        System.out.println(facebook.findMaxOverLapTime(intervalList));
    }

    @Test
    public void findLca(){
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.right.left = new TreeNode(5);
        root.right.right = new TreeNode(6);
        root.left.left = new TreeNode(4);
        System.out.println(facebook.findLcaRecursive(root).val);
    }

    @Test
    public void testTurnValid(){
        System.out.println(facebook.turnToValid("()()aa)d("));
        System.out.println(facebook.balanceParenthese("()()aa)d("));
    }


    @Test
    public void testSubarray(){
        int []nums={-1,-2,3,14,5,6,-5};
        System.out.println(facebook.subArraySum(25,nums));
    }

    @Test
    public void testPrintColumn(){
        PrintByColumn p = new PrintByColumn();
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.right.left = new TreeNode(5);
        root.right.right = new TreeNode(6);
        root.left.left = new TreeNode(4);
        p.print(root);
    }

    @Test
    public void testHasCommonK(){
        facebook.hasCommonThanK("ACEDFFESX","YGUHIJICEDF",3);
    }

    @Test
    public void testbst2double(){
        TreeNode root = new TreeNode(5);
        root.left = new TreeNode(3);
        root.right = new TreeNode(7);
        root.left.left = new TreeNode(2);
        root.left.right= new TreeNode(4);
        root.right.left = new TreeNode(6);
        root.right.right = new TreeNode(8);
        TreeNode link = facebook.bst2DoubleListRecursive(root);
        System.out.println(link.val);
    }

    @Test
    public void testSortByABS(){
        int []nums={-3,-2,-1,2,3,4,5};
        int []ans = facebook.sortedSquares(nums);
        for(int x:ans)
            System.out.println(x);
    }

    @Test
    public void testCoverInterval(){
        Interval []intervals = new Interval[2];
        intervals[0]=new Interval(0,3);
        intervals[1] = new Interval(4,7);
//        intervals[2] = new Interval(4,6);
//        intervals[3] = new Interval(2,7);
        System.out.println(facebook.findCover(intervals,new Interval(0,6)));
    }

    @Test
    public void testFindIntersection(){
        Interval [] A = new Interval[3];
        Interval [] B = new Interval[4];
        /*
        [1,3],[2,6],[8,10],[15,18],
         */
        A[2]= new Interval(10,13);
        A[1] = new Interval(14,16);
        A[0] = new Interval(19,29);
        B[0]= new Interval(2005,2016);
        B[1] = new Interval(2008,2014);
        B[2] = new Interval(2006,2008);
        B[3] = new Interval(2010,2014);
        //facebook.findIntersection(A,B);
        //System.out.println(facebook.findOverlap(B));
        //System.out.println(facebook.meetingRooms(B));
        System.out.println(facebook.insert(new ArrayList<>(Arrays.asList(A)),new Interval(15,20)));
    }

    @Test
    public void testArrange(){
        System.out.println(facebook.arrange("AABACDCD",3));
        StringBuilder sb = new StringBuilder("aced");
        sb.delete(0,2);
        System.out.println(sb.toString());
        System.out.println(sb.toString());
    }
}