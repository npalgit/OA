import commons.DoubleListNode;
import commons.NestedInteger;
import commons.TreeNode;

import java.util.*;

/**
 * Created by tao on 9/4/17.
 */
public class WePay {

    /*
    The first place your computer looks is its local DNS cache, which stores DNS information that the computer has recently retrieved

    Step 2: Ask the Recursive DNS servers


    Finally, the recursive server gives the host record back to your computer.
    Your computer stores the record in its cache, reads the IP address from the record,
    then passes this information to the web browser. Your browser then opens a connection to the IP address '72.14.207.99'
    on port 80 (for HTTP), and our webserver passes the web page to your browser,
     */



    /*
    java的内存管理，gc什么的，lz真的不懂就只能瞎掰
    1. double sqrt（double input）
    2. 电梯design。
    3. nested iterator（leetcode原题）
    4. 详细聊简历1一个小时。问technical很细。

    TCP Sliding Window
    一个设计题，是他自己工作中的内容，类似信用卡validation，讨论了一个小时，我根据理解说了一些可以用的数据结构和算法，写了一些pseudo code. trie
    Implement Hashtable.



    Hiring Manager, 一个文件夹“root”，请找出root底下所有的包含文件名为“src"的文件的路径，值得注意的是，其中有可能包括名字为”src"的文件夹，需要排除在外
    这个我用Trie和queue结合起来做的
    a. 给一个字符串，格式是：xxx IP: 255.255.255.100 include: wepay.com，然后一个文件中包含多个这样的字符串，提取所有的IP地址
   b. 还是这个格式，然后每次遇到xxx.com，提取它的对应的IP地址
   我写了半天，然后小哥好像没怎么认真听，我也觉得我好想没太明白他题目的愿意，最后时间快到的时候他说了个他自己很牛逼的方法。




    问了些基础知识，OS里堆栈的区别，mutex和semephore的区别（第三次被问到了），tcp/udp区别，raid 0123456是什么之类的
     */


    /*
    1. 给你一个BST，把所有的leaf node连成doubly linked list然后return。注意leaf不一定在同一层
2. Leetcode Evaluate Reverse Polish Notation 的变体，非Reverse的Polish Notation，而且输入是一整个string。

Return the number of distinct palindromic substrings for a given string.

2，word break ii, 问复杂度

1)given a string, determine if the string is a valid phone number or not

2)validate an email address

3)given a string of a valid email, make a function that corrects the email address by its doma


灾难总是接踵而至，这正是世间的常理，你以为只要解释一下就有谁会来救你吗？要是死了，就只能证明我不过是如此程度的男人
     */
    //242 valid anagram
    public boolean isAnagram(String s,String t){
        if(s.length()!=t.length())
            return false;
        int m=s.length();
        int []cnt=new int[128];
        for(int i=0;i<m;++i){
            cnt[s.charAt(i)]++;
            cnt[t.charAt(i)]--;
        }
        for(int i=0;i<128;++i){
            if(cnt[i]!=0)
                return false;
        }
        return true;
    }

    public int maxSubArray(int[]nums){
        int maxSum=Integer.MIN_VALUE;
        int curSum=0,n=nums.length;
        for(int i=0;i<n;++i){
            curSum+=nums[i];
            if(curSum>maxSum){
                maxSum=curSum;
            }
            if(curSum<0)
                curSum=0;
        }
        return maxSum;
    }


    //186 reverse words in a string II

    public void reverse(char[]s,int begin,int end){
        while(begin<end){
            char c = s[begin];
            s[begin++]=s[end];
            s[end--]=c;
        }
    }

    public void reveseWords(char[]s){
        int begin=0,end=s.length;
        reverse(s,0,end-1);
        while(begin<end){
            int start=begin;
            while(begin<end && s[begin]!=' '){
                begin++;
            }
            reverse(s,start,begin-1);
            if(begin<end && s[begin]==' ')
                begin++;
        }
    }

    public DoubleListNode getPath(TreeNode root,TreeNode p){
        if(root==null||p==null)
            return null;
        if(root==p)
            return new DoubleListNode(p.val);
        DoubleListNode node = new DoubleListNode(root.val);
        DoubleListNode l = getPath(root.left,p);
        DoubleListNode r = getPath(root.right,p);
        if(l==null && r==null)
            return null;
        DoubleListNode next = l!=null?l:r;
        node.next=next;
        next.prev=node;
        return node;
    }



    public void printDiagnol(TreeNode root){
        if(root==null)
            return;
        Queue<TreeNode> q=new LinkedList<>();
        TreeNode cur=root;
        while(cur!=null || !q.isEmpty()){
            while(cur!=null){
                q.offer(cur);
                cur=cur.right;
            }
            TreeNode top = q.poll();
            System.out.println(top.val);
            cur=top.left;
        }
    }


    //1. 给数组 求平均数: stack overflow
    //2.给数组 求  和是 3的倍数的所有可能的子列（连续） 总数
    /*
    如果一个自数组和是三的倍数 sum[i...j] = sum[0...j]-sum[0...i-1]  那么 sum[0...i-1] 和 sum[0...j]应该是对于3同余的
    所以可以用三个count0 count1 count2 分别代表除以3余0 1 2 的个数，然后 遍历一遍 当计算从0到当前的和 假如余2 那么
    它可以和他前面所有的余2的sum分别组成一个unique的子数组 所以result+=count2 最后得出结果 O(n)
     */

    //brute force
    public int getThreeSum(int[]nums){
        int n = nums.length;
        int []sum=new int[n];
        for(int i=0;i<n;++i)
            sum[i]=(i>0?sum[i-1]:0)+nums[i];
        int cnt=0;
        for(int i=0;i<n;++i){
            if(sum[i]%3==0)
                cnt++;
            for(int j=1;j<=i;++j){
                if((sum[i]-sum[j-1])%3==0)
                    cnt++;
            }
        }
        return cnt;
    }

    //better way
    public int getThreeSumBetter(int[]nums){
        int n = nums.length;
        int cnt0=0,cnt1=0,cnt2=0;
        int cnt=0,sum=0;
        for(int i=0;i<n;++i){
            sum+=nums[i];
            int rem = sum%3;
            switch (rem){
                case 0:
                    cnt+=++cnt0;
                    break;
                case 1:
                    cnt+=cnt1;
                    cnt1++;
                    break;
                case 2:
                    cnt+=cnt2;
                    cnt2++;
                    break;
                default:
                    break;
            }
        }
        return cnt;
    }

    //get all subArray sum by k
    public int subCount(int[]nums,int k){
        int n = nums.length;
        int []cnt=new int[k];
        int cntNum=0,sum=0;
        for(int i=0;i<n;++i){
            sum+=nums[i];
            int rem = sum%k;
            if(rem==0){
                cntNum+=++cnt[0];
            }else{
                cntNum+=cnt[rem];
                cnt[rem]++;
            }
        }
        return cntNum;
    }

    //NestedIterator
    class NestedIterator implements Iterator<Integer>{

        Stack<NestedInteger> q=null;
        public NestedIterator(List<NestedInteger>nestedList){
            q=new Stack<>();
            int n = nestedList.size();
            for(int i=n-1;i>=0;--i)
                q.push(nestedList.get(i));
        }
        public Integer next(){
            return q.peek().getInteger();
        }

        public boolean hasNext(){
            boolean has=false;
            while(!q.isEmpty()){
                NestedInteger top = q.pop();
                if(top.isInteger()){
                    q.push(top);
                    has=true;
                    break;
                }else{
                    List<NestedInteger>li = top.getList();
                    int nn = li.size();
                    for(int i=nn-1;i>=0;--i)
                        q.push(li.get(i));
                }
            }
            return has;
        }
    }

    //4 median of two sorted array

    public int findKth(int[]nums1,int start1,int end1,int[]nums2,int start2,int end2,int k){
        if(end1-start1>end2-start2)
            return findKth(nums2,start2,end2,nums1,start1,end1,k);
        if(start1>=end1)
            return nums2[start2+k-1];
        if(k==1)
            return Math.min(nums1[start1],nums2[start2]);
        int half = Math.min(k/2,end1-start1);
        if(nums1[start1+half-1]<nums2[start2+k-half-1])
            return findKth(nums1,start1+half,end1,nums2,start2,end2,k-half);
        else if(nums1[start1+half-1]>nums2[start2+k-half-1])
            return findKth(nums1,start1,end1,nums2,start2+k-half,end2,half);
        else
            return nums1[start1+half-1];
    }
    public double findMedianSortedArrays(int[]nums1,int[]nums2){
        int m = nums1.length+nums2.length;
        int first = findKth(nums1,0,nums1.length,nums2,0,nums2.length,m/2+1);
        if(m%2!=0)
            return 1.0*first;
        else
            return 1.0*(findKth(nums1,0,nums1.length,nums2,0,nums2.length,m/2)+first)/2.0;
    }


    public boolean isValidSudoku(char[][]board){
        for(int i=0;i<9;++i){
            Set<Character>rows=new HashSet<>();
            Set<Character>cols = new HashSet<>();
            Set<Character>cube = new HashSet<>();
            for(int j=0;j<9;++j){
                if(board[i][j]!='.' && !rows.add(board[i][j]))
                    return false;
                if(board[j][i]!='.' && !cols.add(board[j][i]))
                    return false;
                int rowIndex = 3*(i/3);
                int colIndex = 3*(i%3);
                if(board[rowIndex+j/3][colIndex+j%3]!='.' && !cube.add(board[rowIndex+j/3][colIndex+j%3]))
                    return false;
            }
        }
        return true;
    }
}
