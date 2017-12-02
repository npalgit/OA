package medium;

import commons.ListNode;
import commons.TreeLinkNode;
import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

/**
 * Created by tao on 8/22/17.
 */
public class MediumQuestionsTest {

    MediumQuestions mq =null;

    @Before
    public void setup(){
        mq = new MediumQuestions();
    }

    @Test
    public void lengthOfLongestSubstring() throws Exception {
        System.out.println(mq.lengthOfLongestSubstring("abcabcbb"));;
    }

    @Test
    public void testFindMedium()throws Exception{
        int []arr={3};
        int []arr1={1,2};
        System.out.println(mq.findMedianSorted(arr1,arr));
    }

    @Test
    public void test3Sum()throws Exception{
        int[]nums={-1,0,1,2,-1,-4};
        System.out.println(mq.threeSum(nums));
    }

    @Test
    public void testThreeSumCloest()throws Exception{
        int[]nums={0,2,1,-3};
        System.out.println(mq.threeSumCloest(nums,1));
    }

    @Test
    public void testLetterCombination()throws Exception{
        System.out.println(mq.letterCombinations("23"));
    }

    @Test
    public void testSwapPairs()throws Exception{
        ListNode node = new ListNode(1);
        node.next=new ListNode(2);
        node.next.next=new ListNode(3);
        node.next.next.next=new ListNode(4);
        mq.swapPairs(node);
    }

    @Test
    public void testLongestParentheses(){
        System.out.println(mq.longestValidParentheses("()))(())"));
    }

    @Test
    public void testSearchRange()throws Exception{
        int []nums={1,2,3,3,3,3,4,5,9};
        System.out.println(mq.searchRange(nums,3));
    }

    @Test
    public void testBinarySeach()throws Exception{
        int []nums={1,2,3,4,5};
        System.out.println(mq.binarySearch(nums,6));
    }


    @Test
    public void testMultiply()throws Exception{
        System.out.println(mq.multiply("123","23"));
    }

    @Test
    public void testJump()throws Exception{
        int[]nums={2,3,1,1,4};
        System.out.println(mq.jump(nums));
    }


    @Test
    public void testIsValidNumber()throws Exception{
        System.out.println(mq.isNumber("   "));
    }

    @Test
    public void testSetZeroes()throws Exception{
        int[][]matrix={{1,2,3},{4,5,6},{7,0,3},{0,4,5}};
        mq.setZeroes(matrix);
    }

    @Test
    public void testFulljustify()throws Exception{
        String[]words={"This","is","an","example","of","text","justification."};
        System.out.println(mq.fullJustify(words,16));
    }

    @Test
    public void testConnect()throws Exception{
        TreeLinkNode root = new TreeLinkNode(1);
        root.left=new TreeLinkNode(2);
        root.right=new TreeLinkNode(3);
        root.left.left=new TreeLinkNode(4);
        root.left.right=new TreeLinkNode(5);
        root.right.right=new TreeLinkNode(7);
        mq.connectII(root);
    }

    @Test
    public void testPalindrome(){
        System.out.println(mq.isPalindrome("race a car"));
    }

    @Test
    public void testSolve()throws Exception{
        char[][]board={{'X','X','X','X'},{'X','O','O','X'},{'X','X','O','X'},{'X','O','X','X'}};
        mq.solve1(board);
    }

    @Test
    public void testMinCut()throws Exception{
        System.out.println(mq.minCut("ab"));
    }

    @Test
    public void testInsertion()throws Exception{
        ListNode node = new ListNode(1);
        node.next=new ListNode(3);
        node.next.next=new ListNode(2);
        node.next.next.next=new ListNode(4);
        node.next.next.next.next=new ListNode(5);
        mq.insertionSortList(node);
    }

    @Test
    public void testMaxProduct()throws Exception{
        int []nums={2,3,-2,-4};
        System.out.println(mq.maxProduct(nums));
    }

    @Test
    public void testLargest()throws Exception{
       // int[]nums={3, 30, 34, 5, 9};
        //System.out.println(mq.largestNumber(nums));
        String s="the sky is blue";
        char []ss=s.toCharArray();
        mq.reverseWords(ss);
    }


    @Test
    public void testMaximalSquare()throws Exception{
        char[][]matrix={{'1','0','1','0','0'},{'1','1','1','1','1'},{'1','1','1','1','1'},{'1','0','0','1','0'}};
        System.out.println(mq.maximalSquare(matrix));
    }

    @Test
    public void testCalculate()throws Exception{
        //System.out.println(mq.calculate("(1-(4+5+2)-3)-(6-8)"));
        System.out.println(mq.calculateII("3/2*4/5+    3/4+  3-3+ 2-3 +4-4-3+4*4/5-4"));
    }


    @Test

    public void testDiff()throws Exception{
        System.out.println(mq.diffWaysToCompute("2-1-1"));
    }


    @Test
    public void testPermuat(){
        int[]nums={1,2,3};
        PriorityQueue<Map.Entry<Integer,Integer>>pq=new PriorityQueue<>(new Comparator<Map.Entry<Integer, Integer>>() {
            @Override
            public int compare(Map.Entry<Integer, Integer> o1, Map.Entry<Integer, Integer> o2) {
                return o1.getValue()-o2.getValue();
            }
        });
        Map<Integer,Integer>map=new HashMap<>();
        map.put(1,1);
        pq.addAll(map.entrySet());
        Map.Entry entry=pq.poll();
        System.out.println((int)entry.getValue()<2);
        //mq.permute(nums);
    }

    @Test
    public void testAdditive(){
        //System.out.println(mq.isAdditiveNumber("10111"));
        //System.out.println(mq.removeDuplicateLetters("bcabc"));
        ListNode node = new ListNode(1);
        node.next=new ListNode(2);
        node.next.next=new ListNode(3);
        node.next.next.next=new ListNode(4);
        node.next.next.next.next=new ListNode(5);
        mq.oddEvenList(node);
    }

    @Test
    public void testLongestIncreasingPath()throws Exception{
        int[][]matrix={{7,8,9},{9,7,6},{7,2,3}};
        System.out.println(mq.longestIncreasingPath(matrix));
    }

    @Test
    public void testPatchingArray()throws Exception{
        int[]nums={9, 1, 2, 5, 8, 3};
        //System.out.println(mq.minPatches(nums,6));
        mq.getMax(nums,4);
    }

    @Test
    public void testMaximum()throws Exception{
        int []nums1={3,4,6,5};
        int[]nums2={9,1,2,5,8,3};
        mq.maxNumber(nums1,nums2,5);
    }


    @Test
    public void testValidPreorder()throws Exception{
        System.out.println(mq.isValidSerialization("1,#,#,#,#"));
    }

    @Test
    public void testTreeSet()throws Exception{
        TreeSet<String>q=new TreeSet<>();
        q.add("JFK");
        q.add("ATL");
        q.add("LAX");
        System.out.println("ATL1".compareTo("JFK"));
        for(String str:q)
            System.out.println(str);
    }

    @Test
    public void testFindItinerary()throws Exception{
        String[][]tickets={{"JFK","SFO"},{"JFK","ATL"},{"SFO","ATL"},{"ATL","JFK"},{"ATL","SFO"}};
        System.out.println(mq.findItinerary(tickets));
    }

    @Test
    public void testMatrix()throws Exception{
        int[][]matrix={{1,2,-1,-4,-20},{-8,-3,4,2,1},{3,8,10,1,3},{-4,-1,1,7,-6}};
        System.out.println(mq.getMax(matrix));
    }

    @Test
    public void testIsPerfect()throws Exception{
        System.out.println(mq.isPerfectSquare(16));
    }

    @Test
    public void testPalindromeNumber()throws Exception{

    }

    @Test
    public void testBomnEnemy()throws Exception{
        /*
        0 E 0 0
        E 0 W E
        0 E 0 0
         */
        char[][]matrix={{'0','E','0','0'},{'E','0','W','E'},{'0','E','0','0'}};
        System.out.println(mq.maxKilledEnemies(matrix));
    }

    @Test
    public void testSuperPow()throws Exception{
        int []a={5,0};
        System.out.println(mq.superPow(2,a));
    }


    @Test
    public void testReduence(){
        int[][]edges ={{1,2},{2,3},{3,4},{4,1},{1,5}};
        int []res = mq.findRedundantDirectedConnection(edges);
        System.out.println(res[0]+" "+res[1]);
    }
    @Test
    public void testnextCloseElement(){
        System.out.println(mq.nextClosestTime("23:59"));
    }

    @Test
    public void testMaxA(){
        //1327104
        //8 12
        //9 16
        System.out.println(mq.maxA(50));
    }

    @Test
    public void testPredictParty(){
        System.out.println(mq.predictPartyVictory("DRRDRDRDRDDRDRDR"));
    }

    @Test
    public void isPossible(){
        int [] nums={1,2,3,4,4,5};
        //System.out.println(mq.isPossibleEasy(nums));
        assertEquals(false,mq.isPossibleEasy(nums));
    }

    @Test
    public void testFindDerangement(){
        mq.findDerangement(6);
    }
}