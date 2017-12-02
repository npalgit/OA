import commons.ListNode;
import commons.MultilevelListNode;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by tao on 10/20/17.
 */
public class BloombergTest {

    Bloomberg bloomberg = null;

    @Before
    public void setup(){
        bloomberg =  new Bloomberg();
    }

    @Test
    public void mergeData() throws Exception {
        bloomberg.mergeData();
    }

    @Test
    public void testMoveZerosChange(){
        int []nums={0, 1, 0, 3, 12};
        bloomberg.moveZerosChange(nums);
        for(int x:nums)
            System.out.println(x);
    }


    @Test
    public void testFindKth(){
        int []nums={3,2,1,5,6,4};
        System.out.println(bloomberg.findKthLargest(nums,2));
    }

    @Test
    public void testWordBREAK(){
        List<String>words = new ArrayList<>(Arrays.asList("leet","code"));
        System.out.println(bloomberg.wordBreak("leetcode",words));
    }

    @Test
    public void testReverse(){
        System.out.println(bloomberg.reverseWords("  the    sky   is    blue   "));
        int n = -1,cnt=0;
        while(n!=0){
            cnt++;
            n&=(n-1);
        }
        System.out.println(cnt);
        System.out.println(Integer.bitCount(-1));
    }

    @Test
    public void testJosephus(){
        System.out.println(bloomberg.getLast(5,4));
    }

    @Test
    public void testMimum(){
        int []index = bloomberg.minimumDist("BLOOMBERG",'B');
        for(int x:index)
            System.out.println(x);
    }
    @Test
    public void testSearchTarget(){
        int[]nums={39, 51, 71, 84, 92, 93, 149, 154, 158, 173, 178, 193, 216, 217, 231, 243, 410, 425, 429, 459, 476, 482, 483, 540, 542, 567, 586, 587, 593, 746, 807, 823, 833, 839, 850, 873, 919, 923, 975, 1000, 994, 992, 990, 989, 987, 984, 980, 979, 976, 970, 965, 960, 958, 955, 945, 939, 928, 915, 910, 909, 905, 896, 889, 885, 879, 874, 872, 870, 862, 861, 852, 846, 841, 838, 829, 828, 826, 825, 821, 820, 819, 817, 816, 815, 814, 812, 811, 804, 803, 801, 800, 790, 784, 783, 780, 778, 774, 765, 764, 763, 761, 758, 756, 753, 750, 738, 733, 728, 725, 724, 716, 712, 711, 708, 705, 704, 701, 699, 697, 686, 685, 684, 683, 682, 680, 673, 670, 666, 663, 658, 650, 648, 645, 643, 640, 639, 634, 626, 615, 613, 603, 594, 592, 588, 584, 583, 582, 580, 579, 569, 565, 560, 559, 558, 557, 555, 547, 541, 539, 538, 534, 533, 532, 526, 520, 510, 509, 508, 505, 498, 488, 487, 485, 481, 475, 474, 472, 470, 464, 460, 456, 452, 451, 447, 446, 443, 441, 439, 432, 430, 427, 424, 415, 409, 408, 401, 398, 394, 393, 388, 386, 381, 380, 379, 376, 370, 367, 366, 362, 358, 357, 355, 354, 349, 348, 342, 337, 333, 331, 330, 324, 320, 318, 316, 312, 308, 306, 305, 304, 302, 301, 296, 291, 280, 273, 271, 270, 269, 263, 259, 258, 255, 246, 242, 239, 236, 228, 224, 223, 222, 221, 220, 213, 208, 207, 205, 203, 202, 196, 191, 190, 187, 185, 184, 180, 170, 169, 161, 160, 155, 150, 147, 141, 137, 135, 132, 130, 129, 122, 121, 117, 116, 107, 102, 96, 95, 94, 90, 89, 86, 81, 79, 78, 77, 73, 67, 61, 59, 56, 52, 49, 38, 27, 23, 19, 14, 12, 11, 4};
        //int []nums = {120, 100, 80, 20, 0};
        System.out.println(bloomberg.searchTarget(nums,862));
    }

    @Test
    public void testPrintList(){
        ListNode node = new ListNode(0);
        node.next = new ListNode(1);
        node.next.next = new ListNode(2);
        bloomberg.printListReverse(node);
    }

    @Test
    public void testShortestPalindrome(){
        System.out.println(bloomberg.shortestPalindromeKMP("aacecaaa"));
    }

    @Test
    public void testPrintMultiLevel(){
        MultilevelListNode head = new MultilevelListNode(10);
        head.next = new MultilevelListNode(5);
        head.next.next = new MultilevelListNode(12);
        head.next.next.next = new MultilevelListNode(7);
        head.next.next.next.next = new MultilevelListNode(11);
        head.child = new MultilevelListNode(4);
        head.child.next = new MultilevelListNode(20);
        head.child.next.next = new MultilevelListNode(13);
        head.child.next.child = new MultilevelListNode(2);
        head.child.next.next.child = new MultilevelListNode(16);
        head.child.next.next.child.child = new MultilevelListNode(3);
        head.next.next.next.child = new MultilevelListNode(17);
        head.next.next.next.child.child = new MultilevelListNode(9);
        head.next.next.next.child.child.child=new MultilevelListNode(19);
        head.next.next.next.child.next = new MultilevelListNode(6);
        head.next.next.next.child.child.next = new MultilevelListNode(8);
        head.next.next.next.child.child.child.next = new MultilevelListNode(15);
        //bloomberg.printMultilevel(head);
        //bloomberg.printMultilevelByDepth(head);
        MultilevelListNode node = bloomberg.flatten(head);
        while(node!=null){
            System.out.println(node.val);
            node = node.next;
        }
    }

    @Test
    public void testMove(){
        int []nums={1,-2,-4,5,6,-7,8};
        bloomberg.moveZeros(nums);
        for(int x:nums)
            System.out.println(x);

    }
}
