package hard;

import commons.Point;
import org.junit.Before;
import org.junit.Test;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.*;

/**
 * Created by tao on 10/6/17.
 */
public class HardQuestionsTest {

    HardQuestions hd = null;

    @Before
    public void setup(){
        hd = new HardQuestions();
    }
    @Test
    public void smallestRange() throws Exception {
//        List<List<Integer>>res = new ArrayList<>();
//        res.add(new ArrayList<>(Arrays.asList(4,10,15,24,26)));
//        res.add(new ArrayList<>(Arrays.asList(0,9,12,20)));
//        res.add(new ArrayList<>(Arrays.asList(5,18,22,30)));
//        res.add(new ArrayList<>(Arrays.asList(4,8,8,9)));
//        int[]ans=hd.smallestRange(res);
//        for(int x:ans)
//            System.out.println(x);

        int cnt=0;
        for(int i=1;i<=9;++i){
            if(String.valueOf(i).contains("9")){
                //System.out.println(i);
                cnt++;
            }
        }
        System.out.println(cnt);

    }

    @Test
    public void testMultiplication(){

        //9895
        //28405
        //100787757
        System.out.println(hd.findKthNumber(9895,28405,100787757));
    }

    @Test
    public void testOuterTrees(){
        Point []points = new Point[12];
        points[0]=new Point(0,0);
        points[1]=new Point(0,1);
        points[2]=new Point(0,2);
        points[3]=new Point(1,2);
        points[4]=new Point(2,2);
        points[5]=new Point(3,2);
        points[6]=new Point(3,1);
        points[7]=new Point(3,0);
        points[8]=new Point(2,0);
        points[9]=new Point(1,0);
        points[10]=new Point(1,1);
        points[11]=new Point(3,3);
//        points[12]=new Point(1,4);
//        points[13]=new Point(1,3);
//        points[14]=new Point(1,2);
//        points[15]=new Point(2,1);
//        points[16]=new Point(4,2);
//        points[17]=new Point(0,3);
        System.out.println(hd.outerTrees(points));
    }

    @Test
    public void testMaxSum(){
        Random random = new Random();
//        int []nums=new int[20000];
//        for(int i=0;i<20000;++i){
//            nums[i]= random.nextInt(20);
//        }
        int []nums={1,2,1,2,6,7,5,1};
        int []ans = hd.maxSumOfThreeSubarrays(nums,2);
        for(int x:ans)
            System.out.println(x);
    }


    @Test
    public void testUnique(){
        int [][]matrix ={{0,0,1,0,1,0,1,1,1,0,0,0,0,1,0,0,1,0,0,1,1,1,0,1,1,1,0,0,0,1,1,0,1,1,0,1,0,1,0,1,0,0,0,0,0,1,1,1,1,0},{0,0,1,0,0,1,1,1,0,0,1,0,1,0,0,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,1,0,1,0,0,0},{0,1,0,1,0,1,1,1,0,0,1,1,0,0,0,0,1,0,1,0,1,1,1,0,1,1,1,0,0,0,1,0,1,0,1,0,0,0,1,1,1,1,1,0,0,1,0,0,1,0},{1,0,1,0,0,1,0,1,0,0,1,0,0,1,1,1,0,1,0,0,0,0,1,0,1,0,0,1,0,1,1,1,0,1,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,1}};
        System.out.println(hd.numDistinctIslands(matrix));
    }

    @Test
    public void testFindTrans(){
        //"WRRBBW", "RB"
        System.out.println(hd.findMinStep("RBYYBBRRB","YRBGB"));
    }
}