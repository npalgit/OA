/**
 * Created by tao on 10/20/17.
 */

import com.sun.java.swing.action.HelpAction;
import commons.*;

import java.util.*;


class DoubleListNode{
    public int val;
    public int key;
    public DoubleListNode next;
    public DoubleListNode prev;
    public DoubleListNode(int key,int val){
        this.val=val;
        this.key=key;
    }
}
//class LRUCache {
//
//    DoubleListNode head = null;
//    DoubleListNode tail = null;
//    int _capacity;
//    Map<Integer,DoubleListNode>map=null;
//    public LRUCache(int capacity) {
//        map=new HashMap<>();
//        _capacity=capacity;
//        head=new DoubleListNode(0,0);
//        tail=new DoubleListNode(0,0);
//        head.next=tail;
//        tail.prev=head;
//    }
//
//    public int get(int key) {
//        if(map.containsKey(key)){
//            DoubleListNode node = map.get(key);
//            int val = node.val;
//            //UNLINK and insert it into head
//            insertHead(node);
//            return val;
//        }else
//            return -1;
//    }
//
//    public void insertHead(DoubleListNode node){
//        node.prev.next=node.next;
//        node.next.prev=node.prev;
//        node.next=head.next;
//        head.next.prev=node;
//        node.prev=head;
//        head.next=node;
//    }
//
//    public void put(int key, int value) {
//        if(map.containsKey(key)){
//            DoubleListNode node = map.get(key);
//            node.val=value;
//            insertHead(node);
//        }else{
//            if(map.size()>=_capacity){
//                DoubleListNode deleteNode = tail.prev;
//                deleteNode.prev.next=tail;
//                tail.prev=deleteNode.prev;
//                deleteNode.prev=null;
//                deleteNode.next=null;
//                map.remove(deleteNode.key);
//            }
//            DoubleListNode node = new DoubleListNode(key,value);
//            node.next=head.next;
//            head.next.prev=node;
//            node.prev=head;
//            head.next=node;
//            map.put(key,node);
//        }
//    }
//}
class MinStack {

    private int minVal=0;
    private Stack<Integer>stk = null;
    /** initialize your data structure here. */
    public MinStack() {
        stk = new Stack<>();
    }

    public void push(int x) {
        if(stk.isEmpty()||x<=minVal){
            stk.push(minVal);
            minVal = x;
        }
        stk.push(x);
    }

    public void pop() {
        int top = stk.pop();
        if(top == minVal)
            minVal = stk.pop();
    }

    public int top() {
        return stk.peek();
    }

    public int getMin() {
        return minVal;
    }
}

public class Bloomberg {

    //send data to server

    public static void mergeData(){
        Scanner scanner = new Scanner(System.in);
        Set<Integer>set = new HashSet<>();
        int next = 1;
        while(scanner.hasNext()){
            int val = scanner.nextInt();
            if(val!=next){
                set.add(val);
            }else{
                System.out.println(val);
                next++;
                while(set.contains(next)){
                    System.out.println(next);
                    set.remove(next);
                    next++;
                }
            }
        }
    }




    //leetcode bloomberg tag
    //first reverse the list and then call
    //use stack

    public ListNode reverseList(ListNode node){
        if(node==null||node.next==null)
            return node;
        ListNode newHead = null;
        while(node!=null){
            ListNode next = node.next;
            node.next = newHead;
            newHead =  node;
            node = next;
        }
        return newHead;
    }
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        l1 = reverseList(l1);
        l2 = reverseList(l2);
        ListNode dummy = new ListNode(0);
        ListNode p = dummy;
        int carray = 0;
        while(l1!=null || l2!=null || carray>0){
            carray += (l1!=null?l1.val:0)+(l2!=null?l2.val:0);
            ListNode newNode = new ListNode(carray%10);
            newNode.next = p.next;
            p.next=newNode;
            carray/=10;
            l1=l1!=null?l1.next:null;
            l2=l2!=null?l2.next:null;
        }
        return dummy.next;
    }

    //method 2
    //stack

    //插入到头部，这样就不需要反转结果了
    public void pushIntoStack(Stack<ListNode>stk,ListNode node){
        while(node!=null){
            stk.push(node);
            node = node.next;
        }

    }
    public ListNode addTwoNumbersByStack(ListNode l1, ListNode l2) {
        Stack<ListNode>stk1 = new Stack<>();
        Stack<ListNode>stk2 = new Stack<>();
        pushIntoStack(stk1,l1);
        pushIntoStack(stk2,l2);
        ListNode dummy = new ListNode(0);
        ListNode p = dummy;
        int carry =0;
        while(!stk1.isEmpty()||!stk2.isEmpty()||carry>0){
            carry+=(stk1.isEmpty()?0:stk1.pop().val)+(stk2.isEmpty()?0:stk2.pop().val);
            ListNode newNode = new ListNode(carry%10);
            carry/=10;
            newNode.next = p.next;
            p.next = newNode;
        }
        return dummy.next;
    }

    //o(1) space
    int borrow = 0;
    public ListNode addTwoNumbersRecursive(ListNode l1, ListNode l2) {
        int n1 = 0, n2 = 0;
        ListNode ptr = l1;
        while (ptr!= null) {
            ptr = ptr.next;
            n1++;
        }
        ptr = l2;
        while(ptr != null) {
            ptr = ptr.next;
            n2++;
        }
        ListNode head = null;
        ListNode dummy = null;
        if( n1 > n2)
            head = addTwoNumbers(l1, l2, n1 - n2);
        else
            head = addTwoNumbers(l2, l1, n2 - n1);
        if(borrow == 1) {
            dummy =  new ListNode(borrow);
            dummy.next = head;
            return dummy;
        }
        return head;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2, int diff) {
        if(l1 == null && l2 == null)
            return null;
        ListNode next = null;
        if(diff > 0)
            next = addTwoNumbers(l1.next, l2, diff - 1);
        else
            next = addTwoNumbers(l1.next, l2.next, diff);
        int sum = 0;
        sum += l1 == null ? 0 : l1.val;
        if(diff == 0)
            sum += l2 == null ? 0 : l2.val;
        sum += borrow;
        borrow = sum / 10;
        sum = sum % 10;
        ListNode curr = new ListNode(sum);
        curr.next = next;
        return curr;
    }



    //138 	Copy List with Random Pointer
    Map<RandomListNode,RandomListNode>map = new HashMap<>();
    public RandomListNode copyRandomList(RandomListNode head) {
        if(head==null)
            return null;
        if(!map.containsKey(head)){
            map.put(head,new RandomListNode(head.label));
            map.get(head).next = copyRandomList(head.next);
            map.get(head).random = copyRandomList(head.random);
        }
        return map.get(head);
    }

    //iterative way
    public RandomListNode copyRandomListIterative(RandomListNode head){
        if(head==null)
            return null;
        Map<RandomListNode,RandomListNode>relation = new HashMap<>();
        RandomListNode save = head;
        while(head!=null){
            if(!relation.containsKey(head))
                relation.put(head,new RandomListNode(head.label));

            if(head.next!=null){
                if(!relation.containsKey(head.next))
                    relation.put(head.next,new RandomListNode(head.next.label));
                relation.get(head).next = relation.get(head.next);
            }
            if(head.random!=null){
                if(!relation.containsKey(head.random))
                    relation.put(head.random,new RandomListNode(head.random.label));
                relation.get(head).random = relation.get(head.random);
            }
            head=head.next;
        }

        //save head first
        return relation.get(save);
    }


    //121 best time to buy and sell stock
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int sum = 0;
        if(n<=1)
            return sum;
        int minPrice = prices[0];
        for(int i=1;i<n;++i){
            minPrice = Math.min(minPrice,prices[i]);
            sum=Math.max(sum,prices[i]-minPrice);
        }
        return sum;
    }


    //582 bfs
    public List<Integer> killProcess(List<Integer> pid, List<Integer> ppid, int kill) {
        Map<Integer,List<Integer>>relations = new HashMap<>();
        int n = pid.size();
        for(int i=0;i<n;++i){
            if(!relations.containsKey(ppid.get(i)))
                relations.put(ppid.get(i),new ArrayList<>());
            relations.get(ppid.get(i)).add(pid.get(i));
        }
        Queue<Integer>q = new LinkedList<>();
        q.offer(kill);
        List<Integer>ans = new ArrayList<>();
        while(!q.isEmpty()){
            int top = q.poll();
            ans.add(top);
            List<Integer>children = relations.getOrDefault(top,new ArrayList<>());
            for(int child:children)
                q.offer(child);
        }
        return ans;
    }


    //1 two sum
    public int[] twoSum(int[] nums, int target) {
        int n = nums.length;
        Map<Integer,Integer>index = new HashMap<>();
        for(int i=0;i<n;++i){
            if(index.containsKey(target-nums[i])){
                return new int[]{index.get(target-nums[i]),i};
            }
            index.put(nums[i],i);
        }
        return new int[]{-1,-1};
    }


    //283 moves zeros
    public void moveZeroes(int[] nums) {
        int n = nums.length;
        int i=0,j=0;
        while(i<n){
            if(nums[i]!=0)
                nums[j++]=nums[i];
            i++;
        }
        for(i=j;i<n;++i)
            nums[i]=0;
    }
    //变种是不需要关系顺序的；两次循环就可
    public void moveZerosChange(int[]nums){
        int n = nums.length;
        int i=0,j=n-1;
        while(i<j){
            while(i<j && nums[i]!=0)
                i++;
            while(i<j && nums[j]==0)
                j++;
            if(i<j){
                int tmp = nums[i];
                nums[i++] = nums[j];
                nums[j--]=tmp;
            }
        }
    }


    //20 valid parentheses
    /*
    问如果用户想设定matched pairs, 应该怎么做, . 1point3acres.com/bbs
    比如 想设置,  只关心,  '/' 和 '\' match,  '^' 和 '*' match;  这里假设matched pairs are one-to-one and unique,  比如不存在 '/' 和 '\' match, 并且 '/' 也和 '|' match.
    让我自己设计个函数, 规定输入的参数.
     挺简单的, 我就做了个map 参数,  比如  boolean isValid(String input, Map<Character, Character> map),  用map来做检查
     */
    public boolean isValid(String s) {
        Stack<Character>stk = new Stack<>();
        char []ss = s.toCharArray();
        int n = ss.length;
        for(int i=0;i<n;++i){
            if(ss[i]=='['||ss[i]=='{'||ss[i]=='(')
                stk.push(ss[i]);
            if(ss[i]=='}'){
                if(stk.isEmpty()||stk.peek()!='{')
                    return false;
                stk.pop();
            }
            if(ss[i]==']'){
                if(stk.isEmpty()||stk.peek()!='[')
                    return false;
                stk.pop();
            }

            if(ss[i]==')'){
                if(stk.isEmpty()||stk.peek()!='(')
                    return false;
                stk.pop();
            }
        }
        return stk.isEmpty();
    }


    //117 connect
    public void connect(TreeLinkNode root) {
        TreeLinkNode first = new TreeLinkNode(0);
        TreeLinkNode node = root;
        first.next = node;
        while(first.next!=null){
            node = first.next;
            TreeLinkNode p = first;
            first.next = null;
            for(;node!=null;node=node.next){
                if(node.left!=null){
                    p.next = node.left;
                    p=p.next;
                }
                if(node.right!=null){
                    p.next = node.right;
                    p=p.next;
                }
            }
        }
    }


    //56 merge interval
    public List<Interval> merge(List<Interval> intervals) {
        List<Interval>ans = new ArrayList<>();
        int n = intervals.size();
        if(n<=1)
            return intervals;
        intervals.sort(new Comparator<Interval>(){
            public int compare(Interval i1, Interval i2){
                if(i1.start!=i2.start)
                    return i1.start-i2.start;
                else
                    return i1.end-i2.end;
            }
        });
        ans.add(intervals.get(0));
        for(int i=1;i<n;++i){
            int size = ans.size();
            if(ans.get(size-1).end<intervals.get(i).start){
                ans.add(intervals.get(i));
            }else{
                ans.get(size-1).end = Math.max(ans.get(size-1).end,intervals.get(i).end);
            }
        }
        return ans;
    }

    //sort arrays
    public List<Interval> mergeInterval(List<Interval> intervals) {
        int n = intervals.size();
        List<Interval>res=new ArrayList<>();
        if(n==0)
            return res;
        int []starts = new int[n];
        int []ends = new int[n];
        for(int i=0;i<n;++i){
            starts[i]=intervals.get(i).start;
            ends[i] = intervals.get(i).end;
        }
        Arrays.sort(starts);
        Arrays.sort(ends);
        for(int i=0,j=0;i<n;++i){
            if(i==n-1||starts[i+1]>ends[i]){
                res.add(new Interval(starts[j],ends[i]));
                j=i+1;
            }
        }
        return res;
    }


    public double myPow(double x, int n) {
        if(n==1)
            return x;
        if (n==-1)
            return 1.0/x;
        if(n==0)
            return 1.0;
        return n%2==0?myPow(x*x,n/2):(n<0?1.0/x:x)*myPow(x*x,n/2);
        // if(n==0)
        //     return 1.0;
        // long nn = Math.abs((long)n);
        // double res =1.0;
        // while(nn>0){
        //     if((nn&0x1)!=0)
        //         res*=x;
        //     x*=x;
        //     nn/=2;
        // }
        // return n<0?1.0/res:res;
    }


    //122 best time to buy stock
    public int maxProfitBuyStock(int[] prices) {
        int n = prices.length,sum=0;
        for(int i=1;i<n;++i){
            if(prices[i]>prices[i-1])
                sum+=prices[i]-prices[i-1];
        }
        return sum;

        //Second, suppose the first sequence is "a <= b <= c <= d", the profit is "d - a = (b - a) + (c - b) + (d - c)" without a doubt. And suppose another one is "a <= b >= b' <= c <= d", the profit is not difficult to be figured out as "(b - a) + (d - b')".
        // So you just target at monotone sequences.
    }



    //69
    public int mySqrt(int x) {
        if(x<=1)
            return x;
        // int begin = 1, end =x;
        // while(begin<end){
        //     int mid = (end-begin)/2+begin;
        //     if(mid>46340){
        //         end = mid;
        //         continue;
        //     }
        //     if(mid<=x/mid && (mid+1)>x/(mid+1))
        //         return mid;
        //     else if(mid*mid<x)
        //         begin=mid+1;
        //     else
        //         end =mid;
        // }
        // return begin;

        double first =1.0*x;
        while(Math.abs(first*first-x)>1e-4){
            first = (first+x/first)/2;
        }
        return (int)first;
    }


    //62 unique paths
    public int uniquePaths(int m, int n) {
        if(m<1||n<1)
            return 0;
        // int [][]dp = new int[m][n];
        // for(int i=0;i<n;++i)
        //     dp[0][i]=1;
        // for(int i=0;i<m;++i)
        //     dp[i][0]=1;
        // for(int i=1;i<m;++i){
        //     for(int j=1;j<n;++j)
        //         dp[i][j]=dp[i-1][j]+dp[i][j-1];
        // }
        // return dp[m-1][n-1];


        //O(n) space;
        int []dp = new int[n];
        for(int i=0;i<n;++i)
            dp[i]=1;
        for(int i=1;i<m;++i){
            dp[0]=1;
            for(int j=1;j<n;++j){
                dp[j]+=dp[j-1];
            }
        }
        return dp[n-1];

    }


    //intersection of two linked list
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        //detect whether there is cycle here
        ListNode p = headA;
        ListNode q = headB;
        if(p==null||q==null)
            return null;
        while(p!=q){
            if(p==null)
                p=headB;
            else
                p=p.next;
            if(q==null)
                q=headA;
            else
                q=q.next;
        }
        return p;
    }



    //53 maximum subarray
    public int maxSubArray(int[] nums) {
        int maxSum = Integer.MIN_VALUE, curSum=0;
        int start=0, end=0,newInd=0,n=nums.length;
        for(int i=0;i<n;++i){
            curSum+=nums[i];
            if(maxSum<curSum){
                maxSum = curSum;
                start=newInd;
                end=i;
            }
            if(curSum<0){
                newInd = i+1;
                curSum=0;
            }
        }
        //System.out.println("start "+start+" end "+end);
        return maxSum;
    }

    //7 reverse integer
    public int reverse(int x)
    {
//        int result = 0;
//
//        while (x != 0)
//        {
//            int tail = x % 10;
//            int newResult = result * 10 + tail;
//            if ((newResult - tail) / 10 != result)
//            { return 0; }
//            result = newResult;
//            x = x / 10;
//        }
//
//        return result;
        long xx = (long)x;
        long y = 0;
        while(xx!=0){
            y=10*y+Math.abs(xx%10);
            xx/=10;
            if(y>2147483647)
                break;
        }
        y = x<0?-y:y;
        if(y>2147483647||y<-2147483648)
            return 0;
        return (int)y;
    }


    //287. Find the Duplicate Number
    public int findDuplicate(int[] nums) {

        int n = nums.length;
        for(int i=0;i<n;++i){
            int ind = Math.abs(nums[i]);
            if(nums[ind]<0)
                return ind;
            else
                nums[ind]=-nums[ind];
        }
        return 0;
//        int n =nums.length;
//        int fast = nums[nums[0]];
//        int slow = nums[0];
//        while(fast!=slow){
//            fast = nums[nums[fast]];
//            slow = nums[slow];
//        }
//        fast=0;
//        while(fast!=slow){
//            fast=nums[fast];
//            slow=nums[slow];
//        }
//        return fast;
    }



    //merge two sorted array
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int ind = m+n-1, i= m-1, j =n-1;
        while(i>=0 && j>=0){
            if(nums1[i]>nums2[j])
                nums1[ind--]=nums1[i--];
            else
                nums1[ind--]=nums2[j--];
        }
//        while(i>=0)
//            nums1[ind--]=nums1[i--]; can delete
        while(j>=0)
            nums1[ind--]=nums2[j--];
    }

    //215 findkth largest
    public int quickSelect(int[]nums,int begin,int end){

        int low = begin, hi=end,key =nums[low];//forget here
        while(low<hi){
            while(low<hi && nums[hi]>=key)
                hi--;
            nums[low]=nums[hi];
            while(low<hi && nums[low]<=key)
                low++;
            nums[hi]=nums[low];
        }
        nums[low]=key;
        return low;
    }
    public int findKthLargest(int[]nums,int begin,int end,int k){
        int ind = quickSelect(nums,begin,end);
        if(ind==k-1)
            return nums[k-1];
        else if(ind>k-1)
            return findKthLargest(nums,begin,ind-1,k);
        else
            return findKthLargest(nums,ind+1,end,k);
    }
    public int findKthLargest(int[] nums, int k) {
        return findKthLargest(nums,0,nums.length-1,nums.length+1-k);
    }



    //139 word break

    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String>set = new HashSet<>();
        set.addAll(wordDict);
        int n = s.length();
        boolean []dp = new boolean [n+1];
        dp[0]=true;
        for(int i=1;i<=n;++i){
            for(int j=i-1;j<i;++j){
                System.out.println(s.substring(j,i));
                if(set.contains(s.substring(j,i)) && dp[j]){
                    dp[i]=true;
                    break;
                }
            }
        }
        for(int i=0;i<=n;++i)
            System.out.println(dp[i]);
        return dp[n];

    }



    //reverse words
    public void reverse(char[]ss,int begin,int end){
        while(begin<end){
            char  c = ss[begin];
            ss[begin++]=ss[end];
            ss[end--]=c;
        }
    }
    public String reverseWords(String s) {
        s = s.trim();
        char []ss = s.toCharArray();
        int n = ss.length;
        reverse(ss,0,n-1);
        StringBuilder sb = new StringBuilder();
        int i=0;
        while(i<n){
            while(i<n && ss[i]==' ')
                i++;
            int start =i;
            while(i<n && ss[i]!=' ')
                i++;
            int end =i-1;
            if(end<n){
                while(end>=start)
                    sb.append(ss[end--]);
                sb.append(' ');
            }
        }
        int nn =sb.length();
        return nn==0?"":sb.toString().substring(0,nn-1);

    }


    //3 longest substring without repeating characters
    public int lengthOfLongestSubstring(String s) {
        int n= s.length();
        int len =0 ;
        int begin=0,end=0,duplicate=0;
        int []cnt=new int[256];
        while(end<n){
            if(cnt[s.charAt(end++)]++ ==1)
                duplicate++;
            while(duplicate>=1){
                if(cnt[s.charAt(begin++)]-- ==2)
                    duplicate--;
            }
            len=Math.max(len,end-begin);
        }
        return len;
    }

    //detect cycle
    public boolean hasCycle(ListNode head) {
        if(head==null||head.next==null)
            return false;
        ListNode fast = head;
        ListNode slow = head;
        while(fast!=null && fast.next!=null){
            fast = fast.next.next;
            slow= slow.next;
            if(fast==slow)
                break;
        }
        return fast==slow;
    }



//    public String reverseWords(String s) {
//        if (s == null) return null;
//
//        char[] a = s.toCharArray();
//        int n = a.length;
//
//        // step 1. reverse the whole string
//        reverse(a, 0, n - 1);
//        // step 2. reverse each word
//        reverseWords(a, n);
//        // step 3. clean up spaces
//        return cleanSpaces(a, n);
//    }
//
//    void reverseWords(char[] a, int n) {
//        int i = 0, j = 0;
//
//        while (i < n) {
//            while (i < j || i < n && a[i] == ' ') i++; // skip spaces
//            while (j < i || j < n && a[j] != ' ') j++; // skip non spaces
//            reverse(a, i, j - 1);                      // reverse the word
//        }
//    }
//
//    // trim leading, trailing and multiple spaces
//    String cleanSpaces(char[] a, int n) {
//        int i = 0, j = 0;
//
//        while (j < n) {
//            while (j < n && a[j] == ' ') j++;             // skip spaces
//            while (j < n && a[j] != ' ') a[i++] = a[j++]; // keep non spaces
//            while (j < n && a[j] == ' ') j++;             // skip spaces
//            if (j < n) a[i++] = ' ';                      // keep only one space
//        }
//
//        return new String(a).substring(0, i);
//    }
//
//    // reverse a[] from a[i] to a[j]
//    private void reverse(char[] a, int i, int j) {
//        while (i < j) {
//            char t = a[i];
//            a[i++] = a[j];
//            a[j--] = t;
//        }
//    }




    //always update k
    //k smallest bst
    public int dfs(TreeNode root,int k,int []val){
        if(k<=0||root==null)
            return k;
        k = dfs(root.left,k,val);
        if(k==1){
            val[0]=root.val;
        }
        k--;
        k = dfs(root.right,k,val);
        return k;
    }
    public int kthSmallest(TreeNode root, int k) {
        int []val={0};
        dfs(root,k,val);
        return val[0];
    }


    //8 string to int
    public int myAtoi(String str) {
        //first delete the space
        char []ss =str.toCharArray();
        long sum=0;
        int n=ss.length,ind=0;
        while(ind<n && ss[ind]==' ')
            ind++;
        int sign=1;
        if(ind<n && (ss[ind]=='+'||ss[ind]=='-')){
            sign=ss[ind]=='-'?-1:1;
            ind++;
        }
        while(ind<n && Character.isDigit(ss[ind])){
            sum=10*sum+(long)ss[ind]-'0';
            if(sum>=2147483647)
                break;
            ind++;
        }
        sum=sign*sum;
        if(sum<Integer.MIN_VALUE)
            return Integer.MIN_VALUE;
        else if(sum>Integer.MAX_VALUE)
            return Integer.MAX_VALUE;
        else
            return (int)sum;

    }




    //79 word search
    public boolean exist(char[][]board,int x,int y,boolean[][]vis,int ind,String word){
        if(ind==word.length())
            return true;
        if(x<0||x>=board.length||y<0||y>=board[0].length||vis[x][y]||word.charAt(ind)!=board[x][y])
            return false;
        vis[x][y]=true;

        if(exist(board,x+1,y,vis,ind+1,word)||exist(board,x-1,y,vis,ind+1,word)||exist(board,x,y+1,vis,ind+1,word)||exist(board,x,y-1,vis,ind+1,word)){
            return true;
        }
        vis[x][y]=false;
        return false;
    }
    public boolean exist(char[][] board, String word) {
        //dfs search
        if(board.length==0||board[0].length==0||word.isEmpty())
            return false;
        int m = board.length,n=board[0].length;
        boolean [][]vis=new boolean[m][n];
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(exist(board,i,j,vis,0,word))
                    return true;
            }
        }
        return false;
    }


    //189 rotate array
    public int gcd(int a,int b){
        return b==0?a:gcd(b,a%b);
    }
    public void rotate(int[] nums, int k) {
        int n = nums.length;
        k=k%n;
        int GCD= gcd(k,n);
        for(int i=0;i<GCD;++i){
            int save =nums[i];
            int saveInd=i;
            while(true){
                int ind =(saveInd-k+n)%n;
                nums[saveInd]=nums[ind];
                saveInd=ind;
                if(saveInd==i+k)
                    break;
            }
            nums[saveInd]=save;

        }
    }

    public void rotateSpace(int[] nums, int k) {
        int n=nums.length;
        if(k%n==0)
            return;
        int[]copy=nums.clone();
        for(int i=0;i<n;++i){
            nums[(i+k)%n]=copy[i];
        }
    }
    public void reverse(int[]nums,int begin,int end){
        while(begin<end){
            int tmp=nums[begin];
            nums[begin++]=nums[end];
            nums[end--]=tmp;
        }
    }
    public void rotateThree(int[] nums, int k) {
        int n=nums.length;
        k=n-1-k%n;
        reverse(nums,0,k);
        reverse(nums,k+1,n-1);
        reverse(nums,0,n-1);
    }



    //path sum ii

    public void dfs(List<List<Integer>>res,TreeNode root,int sum,List<Integer>path){
        if(root==null)
            return;
        path.add(root.val);
        if(sum==root.val && root.left==null && root.right==null){
            res.add(new ArrayList<>(path));
            path.remove(path.size()-1);
            return;
        }
        dfs(res,root.left,sum-root.val,path);
        dfs(res,root.right,sum-root.val,path);
        path.remove(path.size()-1);
    }
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>>res = new ArrayList<>();
        dfs(res,root,sum,new ArrayList<>());
        return res;
    }


    //669 trim a bst
    public TreeNode trimBST(TreeNode root, int L, int R) {
        if(root==null)
            return null;
        if(root.left==null && root.right==null){
            return root.val>=L && root.val<=R?root:null;
        }
        if(root.val<L)
            return trimBST(root.right,L,R);
        else if(root.val>R){
            return trimBST(root.left,L,R);
        }else{
            root.left = trimBST(root.left,L,R);
            root.right = trimBST(root.right,L,R);
            return root;
        }
    }


    //24 swap nodes in pair
    //iteartive way
    public ListNode swapPairsIteartive(ListNode head) {
        //iterative way
        if(head==null||head.next==null)
            return head;
        ListNode first=head;
        ListNode second=head.next;
        ListNode dummy = second;
        ListNode pre =null;
        while(first!=null && second!=null){
            ListNode node = second.next;
            second.next = first;
            if(pre!=null)
                pre.next=second;
            first.next=node;
            pre=first;
            first=node;
            if(first!=null)
                second=first.next;
        }
        return dummy;
    }

    //recursive
    public ListNode swapPairs(ListNode head) {
        if(head==null||head.next==null)
            return head;
        ListNode node = swapPairs(head.next.next);
        ListNode first = head.next;
        first.next=head;
        head.next=node;
        return first;
    }


    //friends cycle 并查集和dfs


    //hindex
    public int hIndex(int[] citations) {
        //
        int n = citations.length;
        int []res=new int[n+1];
        for(int i=0;i<n;++i){
            if(citations[i]>=n)
                res[n]++;
            else
                res[citations[i]]++;
        }
        int sum=0;
        for(int i=n;i>=0;--i){
            sum+=res[i];
            if(sum>=i)
                return i;
        }
        return 0;
    }




    public boolean nextPermutation(int[] nums) {
        //
        int n = nums.length;
        int j=n-2;
        while(j>=0){
            if(nums[j]<nums[j+1])
                break;
            j--;
        }
        if(j==-1){
            //alreay the largest
            reverse(nums,0,n-1);
            return false;
        }
        //find the smallest larger one
        int i=n-1;
        while(i>j){
            if(nums[i]>nums[j])
                break;
            i--;
        }
        int tmp = nums[i];
        nums[i]=nums[j];
        nums[j]=tmp;
        reverse(nums,j+1,n-1);
        return true;

    }
    public int nextGreaterElement(int n) {
        char []ss = String.valueOf(n).toCharArray();
        int []nums = new int[ss.length];
        for(int i=0;i<nums.length;++i)
            nums[i]=(ss[i]-'0');
        if(!nextPermutation(nums))
            return -1;
        long sum = 0;
        for(int i=0;i<nums.length;++i){
            sum = 10*sum+(long)nums[i];
            if(sum>2147483647)
                return -1;
        }
        return (int)sum;

    }






    //max difference
    public int maxDiff(int[]nums){
        int n = nums.length;
        int maxDifference = nums[1]-nums[0];
        int minVal = nums[0];
        for(int i=1;i<n;++i){
            if(maxDifference<nums[i]-minVal)
                maxDifference = nums[i]-minVal;
            if(minVal>nums[i])
                minVal = nums[i];
        }
        return maxDifference;
    }



    /* For a given array arr[], returns the maximum j-i such that
       arr[j] > arr[i] */
    int maxIndexDiff(int arr[], int n)
    {
        int maxDiff;
        int i, j;

        int RMax[] = new int[n];
        int LMin[] = new int[n];

        /* Construct LMin[] such that LMin[i] stores the minimum value
           from (arr[0], arr[1], ... arr[i]) */
        LMin[0] = arr[0];
        for (i = 1; i < n; ++i)
            LMin[i] = Math.min(arr[i], LMin[i - 1]);

        /* Construct RMax[] such that RMax[j] stores the maximum value
           from (arr[j], arr[j+1], ..arr[n-1]) */
        RMax[n - 1] = arr[n - 1];
        for (j = n - 2; j >= 0; --j)
            RMax[j] = Math.max(arr[j], RMax[j + 1]);

        /* Traverse both arrays from left to right to find optimum j - i
           This process is similar to merge() of MergeSort */
        i = 0; j = 0; maxDiff = -1;
        while (j < n && i < n)
        {
            if (LMin[i] < RMax[j])
            {
                maxDiff = Math.max(maxDiff, j - i);
                j = j + 1;
            }
            else
                i = i + 1;
        }

        return maxDiff;
    }



    public TreeNode upsideDownBinaryTree(TreeNode root) {
        //recursive way
        if(root==null || root.left==null && root.right==null)
            return root;
        TreeNode leftNode = root.left;
        TreeNode left  = upsideDownBinaryTree(root.left);
        leftNode.left=root.right;
        leftNode.right=root;
        root.left=null;
        root.right=null;
        return left;
    }








    //665 non-decreasing array
    public boolean checkPossibility(int[] nums) {
        int cnt = 0;                                                                    //the number of changes
        for(int i = 1; i < nums.length && cnt<=1 ; i++){
            if(nums[i-1] > nums[i]){
                cnt++;
                if(i-2<0 || nums[i-2] <= nums[i])nums[i-1] = nums[i];                    //modify nums[i-1] of a priority
                else nums[i] = nums[i-1];                                                //have to modify nums[i]
            }
        }
        return cnt<=1;
    }





    //component
    public int countComponents(int n, int[][] edges) {
        UnionFind uf = new UnionFind(n);
        int cnt=n;
        for(int[]edge:edges){
            if(uf.mix(edge[0],edge[1]))
                cnt--;
        }
        return cnt;
    }

    //dfs way to count the components
    public void dfs(int start,boolean []vis,Map<Integer,List<Integer>>neighbors){
        if(vis[start])
            return;
        vis[start]=true;
        List<Integer>neig=neighbors.getOrDefault(start,new ArrayList<>());
        for(int x:neig){
            dfs(x,vis,neighbors);
        }
    }
    public int countComponentsDFS(int n, int[][] edges) {
        Map<Integer,List<Integer>>neighbors = new HashMap<>();
        for(int[]edge:edges){
            if(!neighbors.containsKey(edge[0]))
                neighbors.put(edge[0],new ArrayList<>());
            neighbors.get(edge[0]).add(edge[1]);
            if(!neighbors.containsKey(edge[1]))
                neighbors.put(edge[1],new ArrayList<>());
            neighbors.get(edge[1]).add(edge[0]);
        }
        boolean []vis = new boolean [n];
        int cnt=0;
        for(int i=0;i<n;++i){
            if(!vis[i]){
                cnt++;
                dfs(i,vis,neighbors);
            }
        }
        return cnt;
    }

    //josephus
    public int josephus(int n,int k){
        if(n==1){
            System.out.println(1);
            return 0;
        }
        else{
            int val = (josephus(n-1,k)+k)%n;
            System.out.println(val+1);
            return val;
        }
    }
    public int getLast(int n, int k){
        return josephus(n,k)+1;
    }



    //找到字符串里面每个字符串距离该字符的最短距离
    //min distance
    public int[] minimumDist(String str,char c){
        int n = str.length();
        int []dp = new int[n];
        Arrays.fill(dp,Integer.MAX_VALUE);
        for(int i=0,b=-1;i<n;++i){
            if(str.charAt(i)==c){
                b=i;
                dp[i]=0;
            }else if(b!=-1){
                dp[i]=i-b;
            }
        }

        for(int i=n-1,b=-1;i>=0;--i){
            if(str.charAt(i)==c){
                b=i;
            }else if(b!=-1){
                dp[i]=Math.min(dp[i],b-i);
            }
        }
        return dp;
    }


    //print diagonal of matrix
    //498
    public int[] findDiagonalOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0) return new int[0];
        int m = matrix.length, n = matrix[0].length;

        int[] result = new int[m * n];
        int row = 0, col = 0, d = 0;
        int[][] dirs = {{-1, 1}, {1, -1}};

        for (int i = 0; i < m * n; i++) {
            result[i] = matrix[row][col];
            row += dirs[d][0];
            col += dirs[d][1];

            if (row >= m) { row = m - 1; col += 2; d = 1 - d;}
            if (col >= n) { col = n - 1; row += 2; d = 1 - d;}
            if (row < 0)  { row = 0; d = 1 - d;}
            if (col < 0)  { col = 0; d = 1 - d;}
        }

        return result;
    }


    public static int find(int[]nums,int begin,int end,int target,boolean ASC){
        if(begin>end)
            return -1;
        while(begin<end){
            int mid =(end-begin)/2+begin;
            if(nums[mid]==target)
                return mid;
            else if(nums[mid]>target){
               if(ASC)
                   end=mid;
               else
                   begin=mid+1;
            }
            else{
                if(ASC)
                    begin=mid+1;
                else
                    end=mid;
            }
        }
        return nums[begin]==target?begin:-1;
    }
    public static int searchTarget(int[]nums,int target){
        //find the maximum value first;
        int begin =0, n=nums.length, end = n-1;
        while(begin<end){
            int mid = (end-begin)/2+begin;
            if(mid<=0||mid>=n-1)
                break;
            if(nums[mid]>=nums[mid+1] && nums[mid]>=nums[mid-1]){
                begin = mid;
                break;
            }
            if(nums[mid]>nums[mid+1] && nums[mid]<nums[mid-1])
                end=mid;
            else
                begin=mid+1;
        }
        // to do binary search in both side;
        if(target>nums[begin]||target<Math.min(nums[0],nums[n-1]))
            return -1;
        int ind1 = -1, ind2=-1;
        if(target>=nums[0])
            ind1 = find(nums,0,begin,target,true);
        if(target>=nums[n-1])
            ind2 = find(nums,begin+1,n-1,target,false);
        return Math.max(ind1,ind2);
    }



    //move zeros version
    public void moveZeros(int[]nums){
        int n =nums.length;
        if(n==0)
            return;
        boolean swaped = false;
        while(true){
            for(int i=0;i<n-1;++i){
                if(nums[i]<0 && nums[i+1]>0){
                    swap(nums,i,i+1);
                    swaped = true;
                }
            }
            if(!swaped)
                break;
            swaped = false;
        }
    }

    public void swap(int[]nums,int i,int j){
        int val = nums[i];
        nums[i]=nums[j];
        nums[j]=val;
    }

    public void swap(char[]nums,int i,int j){
        char val = nums[i];
        nums[i]=nums[j];
        nums[j]=val;
    }


    public void replace(char[]chrs,int[]index){
        int n = chrs.length;
        if(chrs==null||n==0)
            return;
        int i=0;
        while(i<n){
            if(index[i]==i)
                i++;
            else{
                swap(chrs,index[i],index[index[i]]);
                swap(index,i,index[i]);
            }
        }

    }


    public void printListReverse(ListNode node){
        if(node==null)
            return;
        printListReverse(node.next);
        System.out.println(node.val);
    }



    //214 shortest palindrome
    public String shortestPalindrome(String s)
    {
        int n = s.length();
        StringBuilder sb = new StringBuilder(s);
        sb.reverse();
        String rev = sb.toString();
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (s.substring(0, n - i) == rev.substring(i))
                return rev.substring(0, i) + s;
        }
        return "";
    }

    public String shortestPalindromeKMP(String s) {
        String temp = s + "#" + new StringBuilder(s).reverse().toString();
        int[] table = getTable(temp);

        //get the maximum palin part in s starts from 0
        return new StringBuilder(s.substring(table[table.length - 1])).reverse().toString() + s;
    }

    public int[] getTable(String s){
        //get lookup table
        int pLen = s.length()+1;
        int[] table = new int[pLen];
        table[0] = -1;
        int k = -1;
        int j = 0;
        while (j <pLen-1)
        {
            //p[k]表示前缀，p[j]表示后缀
            if (k == -1 || s.charAt(j) == s.charAt(k))
            {
                ++k;
                ++j;
                table[j] = k;
            }
            else
            {
                k = table[k];
            }
        }

        return table;
    }


    //42 trapping water
    public int trap(int[] height) {
        //two pointers
        int n = height.length;
        int left=0,right=n-1;
        int sum=0,leftH=0,rightH=0;
        while(left<right){
            if(height[left]<height[right]){
                leftH=Math.max(leftH,height[left]);
                sum+=leftH-height[left++];
            }else{
                rightH=Math.max(rightH,height[right]);
                sum+=rightH-height[right--];
            }
        }
        return sum;
    }

    //239
    public int[] maxSlidingWindow(int[] nums, int k) {
        Deque<Integer>dq=new LinkedList<>();
        //fill in index;
        int n = nums.length;
        if(n==0)
            return new int[]{};
        int []res=new int[n-k+1];
        int ind=0;
        for(int i=0;i<n;++i){
            while(!dq.isEmpty() && nums[dq.peekLast()]<nums[i]){
                dq.pollLast();
            }
            dq.offer(i);
            if(i>=k-1)
                res[ind++]=nums[dq.peekFirst()];
            if(dq.peekFirst()<=i-k+1)
                dq.pollFirst();
        }
        return res;
    }


    //bst to double list node
    public TreeNode bst2double(TreeNode root) {
        if (root == null)
            return null;
        return dfs(root)[0];
    }

    private TreeNode[] dfs(TreeNode root) {
        if (root == null)
            return null;
        TreeNode[] left = dfs(root.left);
        TreeNode head = null;
        TreeNode tail = null;
        if (left != null) {
            head = left[0];
            left[1].right = root;
            root.left = left[1];
        }
        TreeNode[] right = dfs(root.right);
        if (right != null) {
            root.right = right[0];
            right[0].left = root;
            tail = right[1];
        }
        TreeNode[] ret = new TreeNode[2];
        ret[0] = (head == null ? root : head);
        ret[1] = (tail == null ? root : tail);
        return ret;
    }

    //114 flatten binary tree to linked list
    TreeNode pre =  null;
    public void flatten(TreeNode root) {
        if(root==null)
            return;
        flatten(root.right);
        flatten(root.left);

        root.right=pre;
        root.left=null;
        pre = root;
    }


    //flatten a multi-level linked list
    //not in depth
    public void printMultilevel(MultilevelListNode head){
        Queue<MultilevelListNode>q = new LinkedList<>();
        if(head!=null)
            q.offer(head);
        while(!q.isEmpty()){
            MultilevelListNode top = q.poll();
            for(;top!=null;top=top.next){
                System.out.println(top.val);
                if(top.child!=null)
                    q.offer(top.child);
            }
        }
    }

    public void printMultilevelByDepth(MultilevelListNode node){
        if(node==null)
            return;
        System.out.println(node.val);
        printMultilevelByDepth(node.child);
        printMultilevelByDepth(node.next);
    }

    MultilevelListNode previous=null;
    public MultilevelListNode flatten(MultilevelListNode node){
        if(node==null)
            return null;
        MultilevelListNode next = node.next;
        previous = node;
        if(node.child!=null)
            node.next = flatten(node.child);
        if(previous!=null && next!=null)
            previous.next = flatten(next);
        return node;
    }



    //*2 /3 path number
    //*2/3
    public static int pathBFS(int start, int target) {
        if (start == target) {
            return 0;
        } else if (start == 0) {
            return -1;
        } else if (start * target < 0) {
            return -1;
        }
        HashSet<Integer> visited = new HashSet<>();
        Queue<Integer> q = new LinkedList<>();
        visited.add(start);
        q.offer(start);
        int count = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                int cur = q.poll();
                if (cur  == target) {
                    return count;
                }
                if (cur == 0) {
                    continue;
                }
                if (!visited.contains(cur * 2)) {
                    visited.add(cur * 2);
                    q.offer(cur * 2);
                }
                if (!visited.contains(cur / 3)) {
                    visited.add(cur / 3);
                    q.offer(cur / 3);
                }
            }
            count++;
        }
        return count;
    }



    //top k frequent 692
    class Tuple{
        public String word;
        public int fre;
        public Tuple(String word,int fre){
            this.fre = fre;
            this.word = word;
        }
    }

    public int comp(String word1,String word2){
        int i =0 , j=0, m=word1.length(), n = word2.length();
        while(i<m && j<n){
            if(word1.charAt(i)!=word2.charAt(j))
                return word1.charAt(i)-word2.charAt(j);
            i++;
            j++;
        }
        if(m==n)
            return 0;
        if (m<n)
            return -1;
        return 1;
    }
    public List<String> topKFrequent(String[] words, int k) {
        Map<String,Integer>map = new HashMap<>();
        List<String>ans = new ArrayList<>();
        for(String str:words){
            int val = map.getOrDefault(str,0);
            map.put(str,val+1);
        }

        PriorityQueue<Tuple>pq = new PriorityQueue<>(new Comparator<Tuple>(){
            public int compare(Tuple t1, Tuple t2){
                if(t1.fre!=t2.fre)
                    return t1.fre-t2.fre;
                else
                    return -comp(t1.word,t2.word);
            }
        });

        for(Map.Entry<String,Integer>entry:map.entrySet()){
            String word = entry.getKey();
            int val = entry.getValue();
            if(pq.size()<k){
                pq.offer(new Tuple(word,val));
            }else if(pq.peek().fre<val){
                pq.poll();
                pq.offer(new Tuple(word,val));
            }else if(pq.peek().fre==val && comp(pq.peek().word,word)>0){
                pq.poll();
                pq.offer(new Tuple(word,val));
            }
        }
        while(!pq.isEmpty()){
            ans.add(pq.poll().word);
        }
        Collections.reverse(ans);
        return ans;

    }




    public int[] nextGreaterElement(int[] findNums, int[] nums) {
        int n = findNums.length,m=nums.length;
        int []res=new int[n];
        //先一个map建起来
        Map<Integer,Integer>map=new HashMap<>();
        Stack<Integer>stk=new Stack<>();
        for(int i=0;i<m;++i){
            while(!stk.isEmpty() && stk.peek()<nums[i]){
                map.put(stk.pop(),nums[i]);
            }
            stk.push(nums[i]);
        }
        Arrays.fill(res,-1);
        for(int i=0;i<n;++i){
            if(map.containsKey(findNums[i]))
                res[i]=map.get(findNums[i]);
        }
        return res;
    }
    


    //leader board
    //hashcode function
    //object建了会发生什么
    // oop 的mian concept
    //inheritence 和 polymorphism 的difference

    //t is the mechanism in java by which one class is allow to inherit the features(fields and methods) of another class
    /*
    array and list difference
    implement hashset, and hashcode
     */

    //朋友身高的那道题, 两次bfs
    //flatten a multi-level linked list
    //hit counter




    //stack overflow, how to write a garbage collector in C++ or java, where is reference created;

    //isSumK
    //number of distinct substring: suffix array

    /*
    design，赛跑题，每个选手到一个打卡点，就update pair（选手NO#，打卡点NO#），要在任意时间点pause时，都能打印出所有选手的排名。
    好像lc有原题把，但我没做过，现场想到了double linked list（好像那题就这么做）。
    亚裔姐问了个谜之题目，六边形的蜂巢，每个node有六个相邻的node，求起点到终点的路径，不要求写code。答DFS。

    solution 2：基于double linked list, hashmap
    1. 建立一个HashMap<Runner, Node>, 每个runner对应一个node，O(1)时间找到这个此runner的位置
    2. 每个sensor都建立double linked list，O(1)时间删除，且始终有序。删除后加入更新的sensor链尾即可，update时间O(1)
    3. 建立另一个长 k 的 double linked list，每个node代表一个sensor，将稀疏的sensor连起来
    4. getRank(k)时，依次将每个sensor node中选手按许倒出即可，O(k)



    大叔随手拈来道设计题，没让写code说想法。给一个heap类的两个interface，
    实现heap collection的几个方法，lz heap用的比较少刷题除了priorityheap也没怎么用过


    后面的设计题是股票那个（公司，股价）求avg用什么data structure，top n用什么。哦，一个小知识点，小哥问java里对应c++的指针叫什么，我说java没有显性指针，他说没事了。结果回来查应该是reference？
    白板一遍 地里面经看了几十个（设计题万变不离其宗）why bb类似的behavior question针对自己的背景准备了一下


    印度小哥一枚 在trading组做.

    问了一下简历 , 有哪些比较changeling的部分, 你是怎么克服的.

    然后考了一个 flatten linked list LeetCode类似的题目.

    2种做法, 一个用stack的 一种dfs的方法. 两种都问了.

    然后时间复杂度是多少.

    下午收到了onsite的消息.
    感谢不杀之恩



http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=298773&extra=page%3D7%26filter%3Dsortid%26sortid%3D311%26sortid%3D311
先behavior问为啥bb，讨论简历，然后两道题，第一题买卖股票
buy $15/100股 $13/50股 $10/60股
sell $16/200股 $12/60股 $9/50股
要算总共可以成交最多多少个share，$15/100shares意思是，单价15，想买100股有个问题是如果有人愿意用15去买，
那他肯定愿意用13去买，如果有人愿意用16单价卖，肯定愿意用20单价去卖
比如15块那个家伙，可以用12的价格去卖60股，50的价格卖50股，但是他只想要100股，所以只成交100股，只需要考虑一次交易，不用考虑已经被买了后面就没了
第二题是一些人，每个人有个日历，这天有空或者没空，找到哪天有空人最多

二十分钟前，最不爽的一次面试。格子题，给起点终点，格子里面放数字，比如5，那么踩到这儿以后，还能继续踩的数量，比如在终点前走了超过5步就原地爆炸了，问能否到终点
可能我是最后一个面试，面试官极其push，说一句话他说十句，不给时间好好思考，而且全程没反馈，一直看表，我去年买了个表给你看。虽然写完了，但是估计是跪了

走到5以后，再走一步1就爆炸了，因为这是走到1以后走出的第二步，超过了1

就是买方会愿意用比自己的低价买，卖方愿意用比自己的高价卖，但是必须是卖出数量和买入数量的最小值


http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=295598&extra=page%3D7%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

2. 设计题, 给一个文件, 上面有很多条log, log的形式是: "节目|集数|用户id", 定义一个标准x, 表示至少70%看过前x集的用户接着看完所有的集数, 输出每个节目对应x的最小值.
- 如果一个节目不存在这个x, 则不用输出.
- 一个用户如果看了i集, 那么, 第1到第i集都会看, 比如用户看了第三集, 那么文件中log就会出现, "节目|1用户id", "节目|2用户id", "节目|3用户id", 不会出现只看第三集的情况.
- 假设每个节目有10集.

要设计数据结构, 然后完成process(show, episode, id), print_x()




http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=298955&extra=page%3D7%26filter%3Dsortid%26sortid%3D311%26sortid%3D311
两个面试官，一个做内部搜索系统的，一个做类似worldquant的websim的。一开始简历讲的很兴奋，结果面试题没有算法题，只有一个设计题，
让我设计一个接口和数据结构来存储所有的简历，要求简历每一栏都要分开存储，并且能够高效的搜索符合每一栏当中内容的简历。




http://www.1point3acres.com/bbs/thread-298907-1-1.html

第一轮算法，一切都想着美好的方向发展。
第一题：二维数组，行列的方向都是递增，然后随机把数组中间的一个数换成一个新数，问新数组是否还符合原来的要求。然后把二维数组改成二维LinkedList，同解（需要用recur跟interative两种方法写），没问题。
第二题：写一个存股票的数据结构，股票input={股票名，股票数量，单支价格}，例如{AAPL，50，$100}，要求按照股票名分类，按照股票价格排序，楼主的做法是HashMap<股票名,TreeMap<股票价格,ArrayList<股票数量>>>。然后问买N股股票，一共需要花多少钱。然后scale up，楼主给出用cache的方法，过关。
第二轮OOD，你一定会这么认为的，对吧。楼主在这里膝盖都要跪碎了。
美国大哥（10+BB，SRE方向），美国大哥（10+BB，Communication Infra方向）
进门寒暄：大哥1一分钟，大哥2一分钟。轮到我介绍了，对吧，NoNoNo，咱们直接做题吧。
. 鐗涗汉浜戦泦,涓€浜╀笁鍒嗗湴
第一题：cat这个命令在Linux环境下的意义是什么。把两个文件连一起？No。10分钟后，告诉我，是把第一个文件的stdout直接pipe到第二个文件的stdin。（好吧，楼主也是个明白人，知道onsite肯定是跪了）。接着让我写两个C程序（特地说，我不让你写C++哦），和一个commandline的命令，让第一个C程序的结果直接作为第二个C程序的input。我。。。是去年刚转CS的。。。呵呵。
第二题：Inter-process data commmunication。楼主旁听过基础的操作系统，知道只有Shared Memory，Socket，data pipeline,大哥1一个劲儿的说不够不够，但楼主真的不会了。然后让我每种方法都写一个例子出来。。。。不会。
第三题：他们这是也肯定是知道我不合适了，说咱聊聊你做过的project吧。当时已经精疲力尽了，心想爱咋样咋样吧。楼主之前干过一个IoT的小project，让我介绍那个。我是自个儿一行一行码的几千行代码，自然这个都是知道的。高潮来了，大哥从system reliability的角度(我就一个project，又不是BB家的服务器啊)，批评了我的project如何没有考虑crash recovery，和single node failure，应该怎么怎么去优化。我第一次听说，还有面试官逐条批评project的，涨知识了。
然后就被领出来了。
总结：跪的冤枉不冤枉？不冤枉，那些计算机的基础知识，我真的只是知道皮毛，以后得多往哪个方向学习，多去旁听本科的课程。跪的亏不亏？亏，楼主看了从9月到10月至少30篇BB的new grad面经（感谢地里的朋友），没有一个面经问道过操作系统这个级别的，最多也就是JVM里面的heap stack之类的（楼主都有认真的准备）。这次经验就算是为了5年以后找senior职位的时候提个醒吧，计算机基础很重要。




http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=296295&extra=page%3D9%26filter%3Dsortid%26sortid%3D311%26sortid%3D311


过了大约一个礼拜收到了onsite通知，原本是希望on campus第二轮，无奈bloomberg当时不在学校，于是安排让我去纽约onsite，之前看地里说2轮挂4轮过，不知道3轮是什么。。。
第一轮：技术面，两道题，第一道是k smallest elements from BST(leetcode原题),讲完思路，面试官画了个testcase让走一遍代码，第二题是word square (leetcode 425), 刷了以前bb的面经没遇到过，不过幸好以前做过，所以还是答出来了，要求现场写代码
第二轮：hr面，主要问了问behavior questions,以及用浅显易懂的语言介绍一个简历里面做过的project，还有一些常规问题，比如薪水，期待之类. Waral 鍗氬鏈夋洿澶氭枃绔�,
第三轮: manager面，没有问coding，还是一下behavior question，并且介绍了一下自己做了什么






http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=297893&extra=page%3D9%26filter%3Dsortid%26sortid%3D311%26sortid%3D311



Ps:第一次发面筋结果码号字提交404了。重来码了一遍
再次提交截了图结果504了……想发图片发现上传图片不能就又去找了个image convert to text 的软件。
所以本来热乎的面经被凉到现在……恩……这是份虔诚的面经。

纪念秋招第一次电面。一个native小哥全程很nice感觉。先聊了10几分钟简历，感觉自己说的不是很好。然后开始tech。他说会有算法相关的东西，他可以不注重正确性但是重视逻辑。然后直接就问你对什么数据结构比较 悉？我Array, Set？他说然后呢，我说HashMap？他说然后呢？我说Stack Queue？他说我们来实现一个 queue吧！我先写了queue的几个method的框架add, remove, size, isEmpty,然后在new LinkedList的时候 他说我们先实现一个LinkedList吧！然后开始写LinkedList磕磕绊绊写完了。他说后面queue的框架就不用填了很简单，但是用ArrayList和Linked List实现有什么差别呢？我说了几条一直没答到他说的点上，然后他就引导我看node的定义，说有个next。然后终于想起来，LinkedList每add一个node都要allocate一次内存，而 ArrayList只有在扩容的时候allocate once,所以这一点上Linked List比较expensive.

将近40分钟过去了，我以为小哥要弃疗的时候他又丢给我一个easy, integer to string。说了思路之后写了一 下，还有个小bug。在一起过testcase的时候被我发现了然后改了过来。然后就问了小哥几个问题大概不到5分钟。整个面试过程好像有50分钟多点。 好像是3-5工作日出结果，我这个情况是不是基本快挂了呀……

通过身边朋友的反馈感觉他家面的比较灵活而且个体差异比较大，有人面到两道easy有人只有一个hard，有些题是利扣变种而且不是他家tag下的。所以还是多多刷题吧!
水水的我求rp,希望在水深火热的秋招季大家offer多多！

就是给个int然后print出来，都不用考虑sign什么的， 我就是一个while然后append到str上再print的




//http://www.1point3acres.com/bbs/thread-297966-1-1.html

On campus interview 一般都是 面完之后会当场告诉你结果，如果第一轮你面过了，他会给你约第二轮时间，如果第二轮 过了 会跟你说 接下来约senior manager和 HR面试。
第一轮tech interview 两道题目： 第一道题目给你一个字符串，和一个字符，让你找到这个字符串里面每个字符距离该字符的最短距离。比如 BLOOMBERG 和字符B ，返回数组[0,1,2,2,1,0,1,2,3]. 比较简单的写法是 从左边扫一遍 从右边扫一遍，更新一个数组就好了。我当时写的每次碰到新的B则返回去更新array，写的比较复杂，给三姐面试官解释了好半天。。。
第二道题目 也不难 给你一堆人从纽约飞各个地方开会的cost，比如 A 去 城市SF， LA的cost 200，300， C去城市SF， LA的cost 100， 400，  D去城市SF， LA的cost 320， 210。。。然后保证一半的人去SF 一半的人去LA，使得总的cost最小

第二轮是LRU的变体：一个马拉松比赛，假设路上有10个marker，然后你需要设计几个函数 Top(k) 返回跑在前面的k个人的id， Update（runnerId，markerId）每次跑到某个marker的时候 call这个函数。Hashmap + linkedlist-google 1point3acres

HR面半个小时，大概就是讲一讲Why bbg，why you？还有你选择公司的top3的特质，你是怎么听说bbg的。-google 1point3acres

Manager面试一个小时，问了一道很简单算法（都不能称得上算法题吧）。。。一个string，返回第一个只出现一次的字符。先说用HashMap, manager 说不能用hashmap，然后就改成了一个统计词频的数组，扫两遍。其他时间大部分是manager告诉你bbg的terminal是个什么样的东西，bbg在他眼里如何是个community rather than 卖news 和report的公司。



http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=297313&extra=page%3D13%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

上来问你favirote project，然后你在project里面怎么和队友resolve conflict

然后做题
1. print  diagonal of matrix
2. valid anagram
3. design system to get top 8 url in browser. 就是两个function， getHit(string url) 和getTop8()


http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=298071&extra=page%3D13%26filter%3Dsortid%26sortid%3D311%26sortid%3D311


就面了两轮， 面了我三个小时
第一轮： 一个三哥 一个白人， 一上来问的是 简历和项目， 然后出了两个题目：
第一题 word break 2 但是只找一个答案， 我说了用一个boolean array 做pruning， 那三个一定要说我不对，然后不停的给我hint，让我用hashmap， 然后我说用index 作为key， value就是搜索到的结果。
然后他说 你用index做key， 不直观，要用substring， 牛了个逼。  估计这三个也就背了那一个答案
第二题，是 LeetCode的原题，就是 一个party 人来人走，最后随机找个人 具体题号忘记了

第二轮： 两个三哥哥
第一题 是system design， 揪着我的项目不停问，问我怎么scale 什么的
第二题 是两个 sorted array 找第K 个元素， lgn 解， 我挂在这里了， base case 实在是搞不出来了



http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=297802&extra=page%3D13%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

第一轮：1 给你一大堆log有开始和结束点，每个log相当于一个进程。然后让你design一个东西可以支持输入一个时间点，返回这个时间点有多少个进程在跑。
2. 第二题怎么都想不起来了。。。应该是不是很难，不然会记得，好像就是一个DFS还是BFS的东西。。。

第二轮：
1. 伪装版的LRU。让全部写出来。。。然后follow up太多log怎么办，回答数据库，分布存储，+++
2. 输入一个个pair，pair（A, B） 意思就是A比B大。初始化结束后，给你两个字符，返回他们的关系
. more info on 1point3acres.com
第三轮：.1point3acres缃�
聊简历，一个senior manager面的
-google 1point3acres
第四轮：. Waral 鍗氬鏈夋洿澶氭枃绔�,
HR面试，聊pending offer + package

时间线：. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷
10/03/2017 第一轮
10/04/2017 第二轮. From 1point 3acres bbs
10/06/2017 第三第四轮
10/12/2017 收到据信。

最近比较衰。。。发面经求人品TTT





http://www.1point3acres.com/bbs/thread-179653-1-1.html


第一次发帖好紧张

我是上周去NY onsite的， 面了四轮。看过BB家面经的都应该大致知道的，就不重复啦，贴题：

Round 1:
－ Given an array of n elements， 已知所有elements都在1-n里面，并且有只有一个数字missing，一个数字duplicate。要求return那个duplicate。 Example：［1，3，4，3］return 3。 Linear time，constant space （是Leetcode题吗？ 是的话大家直接无视我吧，我题刷的少。。）.鐣欏璁哄潧-涓€浜�-涓夊垎鍦�

－ Leetcode merge intervals

Round 2:
－ 定义了一个 structure Node { int val; Node* up; Node* down; Node* next;  }; 以及一系列的rules：个人觉得这题的重点是理解rules）。。
          1. 一个Node的 up如果不是NULL，那么那个up Node的down和next都是NULL；
          2. 一个Node的 down如果不是NULL，那么那个down Node的up和next都是NULL；
          3. 所有能通过up pointer reachable的Nodes 的value < current Node‘s value；
          4. 所有能通过down pointer reachable的Nodes 的value > current Node‘s value；
          5. 所有能通过next pointer reachable的Nodes 的value > current Node‘s value and thoses of all Nodes' reachable from down pointer；

  然后given head node， print out the values in the linked list in ascending order。。

  － 然后design题：N个运动员，K个checkpoints， 每个人经过一个checkpoint的时候(runner_id, checkpt_id)就会被加进某data structure里， 问什么样的data structure can make it easy to get the L (L provided by user) leading runners.

Round 3: HR... 不多说了

Round 4: Manager
因为manager自己在做news classification， 所以就聊了各种machine learning的知识。。

嗯 大概就这样了。。今天去follow up后得到电话offer了（找工半年来第一个offer啊），来这里报个到希望对大家有帮助。




http://www.1point3acres.com/bbs/thread-298512-1-1.html

上周三Career Fair投的简历, 上周四schedule了今天第一轮on campus。第一轮面完之后面试官马上问我今天晚点有没有时间，然后约了一个小时之后的二轮（不得不说BB效率真是太高了。。。）。两轮都是先聊简历（主要问实习经历），算法题，然后Q&A。面试官全程很nice。
第一轮：第一题给一个先降后升的array，找一个数，return boolean，我的解法是用两次binary search. 第二题给一个string, 找第一个不重复的character。
第二轮：LC127

祝大家好运！！求BB onsite求亚麻onsite求offer！！










http://www.1point3acres.com/bbs/thread-298269-1-1.html

第一轮。两个面试官，一个在BB家工作11年，另一个9年。一人出了一道题。第一题是给了个二维的linkedlist然后让我search一个target。第二题是处理字符串，很简单。但是follow up不让用任何split(),trim()等一系列自带的function。。。 面完了我问如果有后续环节，下一步是什么？然后11年那个面试官让我出去等，他们要商量一下。过了一会他出来和我安排了下一轮面试。
第二轮。一个面试官。问了两道题。第一个类似于LRU的变形，答得磕磕绊绊，但算是写出来了。第二个是实现一个function 输入一个string 一个char 然后让你找这个string里面所有字母到这个string里含有的这个char的位置的最近的距离。输出一个array。答得也很一般。。。（感觉最差的就是这轮了）。。。面完他给约了第三四轮（连着的，一个senior manage，一个hr）。

第三轮 一个工作28年掌管500人的senior manage。聊简历，十分详细的聊。问了一个LC387。之后又全程尬聊。。。感觉聊的是挺开心的，但是不清楚，他考察的点是什么。。。所以现在很后怕，怕自己哪里没回答好。

第四轮 hr。聊pending offer，why bb，选工作时最看重的因素等等。也是同样不知道hr关注的点是什么，所以现在也很后怕。

我本以为可能过了前两轮也许是成功的一半？尤其是第一轮还让我回避他们讨论。。。但感觉好像并不是。。。可能每个人都会面满四轮吧。。。因为hr说要综合四轮的feedback最终决定要不要你。。。
. 鍥磋鎴戜滑@1point 3 acres
总体来说 他家的面试官很棒，人都很好。安排的也很高效，密集。据说下周就要出结果了。。。
所以发个面经攒攒人品啊！！！希望能有好运。

以及他家是在纸上出题，写题。。。我写的字超级乱。。。所以我能给的建议就是，要提前练习一下在白纸上写code的能力。。。

求offer啊求offer啊求offer啊



http://www.1point3acres.com/bbs/thread-276411-1-1.html

move zeros (but change zeros to odd numbers)1. no order2. odd number keep order
3. both even and odd keep order

求rp 求onsite








http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=291659&extra=page%3D17%26filter%3Dsortid%26sortid%3D311%26sortid%3D311


面试官印度哥哥，还比较nice， 问了fullstack 的一个项目经历，更感兴趣服务器端的架构。

第一题是LC139 的变种，要求output 一个space 分割开每个dictionary word的string，比如：
input: "leetcode", { "leet", "code" }
output: "leet code"
复制代码
第二题没有见到过，请做过的同学帮忙点明出处。
给两个vector A 和 B， A 包含字母，B 包含integer， 把A 按照B 的顺序重新组合，要求不能申请额外的空间，比如：
input: { 'a', 'b', 'c', 'd' }, { 3, 1, 0, 2 }
output: { 'd', 'b', 'a', 'c' }
复制代码

补充内容 (2017-9-7 08:02):
忘了补充说明第二题时间复杂度要求O(n) linear time
. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷
补充内容 (2017-9-8 19:05):
楼下csprogramming 的解法很好理解，就是说在swap A 里character 的时候同时swap B 里面对应的position，这样可以保持A， B 同步update


http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=291616&extra=page%3D17%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

lz新人，初来乍到，第一次发帖来给地里贡献一个去年的bloomberg实习面试跪经。
简历是海投的，大概一个月后收到hr邀约。后来hr发信来通知lz他家实习名额满了，不过之后表示他们已经发出的offer或许还有被拒的可能，说不定我还可以作为替补，问我还愿不愿意继续面，我同意了。

面试题目有两道。第一道是lc的merge 2 sorted arrays。当天或许是心情紧张的缘故，我一上来给的答案是把两个数组append在一起，然后给新数组排序。。。面试官当然不太满意这个答案了，于是给我提示两个数组本身已经排序好了，问我如何利用。然后愣是半天没想出来。。。最后还是面试官点出可以用two pointers才解决的。orz

第二道就是经典的two sum，不过这时面试时间也快到了，面试官就让我口头描述一下思路。描述完了，问了一点他家公司的问题，面试结束。

果不其然还是拒了，通过这次面试，得到的经验就是，刷题才是王道。。。而且刷完后一定得总结归纳，lz当时就是因为第一次找实习，刷题准备不充分才导致这样的结果。

虽然面经是去年的，不过希望能够对今年投bloomberg的同学们起到帮助，祝大家好运！






http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=298033&extra=page%3D17%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

BB onsite Oct. 12, 2017
First round:
3个面试官, 一个国人, 一个印度人, 一个shadow
Q1. Top-K elements in a stream, and distributed top-k.
       LZ刚开始比较紧张, 出了个很尴尬的bug, 被指出来了.
Q2. Word Break.

Second round:
一个面试官, 看起来像国人.
Q1. 设计一个股票系统,  要两个function,
       getAverage()  能指定frame的大小.
       .... 另外一个想不起来了....
Q2.  给一批人的Year-of-birth, year-of-dead. From 1point 3acres bbs
       求在哪一年中活着的人数最多..
Third round:
大boss, 11+ 年BB工作经验.  就聊了下project experience.

然后HR就给我今天的面试结束.....
很奇怪, 没有HR面试.
之前看面经, 要不2轮(fail), 要不4轮, 好像没看到有3轮的情况.
不过目测还是要fail了.






http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=297450&extra=page%3D17%26filter%3Dsortid%26sortid%3D311%26sortid%3D311




学校CF投的简历，然后学校面试~

第一轮，简历问的特别特别详细，但是题目比较简单，move zeros和add two numbers，出来的时候很开心，觉得可以去纽约玩一圈了· 面完收到通知第二天面第二轮。。。。. more info on 1point3acres.com


第二轮，自我介绍了下就开始做题了，第一题是DP，类似俄罗斯套娃那个题，当时也没想明白反正最后也没做出来。第二题是LRU的思路，不用写代码，讲一下就好了，估计是因为另外一个小哥见我很沮丧再加上时间也快到了。这轮结束觉得已经挂了。结果傍晚收到通知第二天面第三轮。

第三轮，和第一个（忘了问他是干啥的了&#128531;）就聊了聊简历，做了个超级超级简单的卖股票的题，也没问什么设计题。然后又和HR去聊了下，各种behavior问题。这一轮感觉见了两个HR。。

第二周收到拒信。。。连续面三天，中间还有一个巨难的期中考试，忧伤




http://www.1point3acres.com/bbs/thread-292767-1-1.html

校园招聘会跟他们聊得不错，晚上就收到了面试邀请。
第二天面试邀请，上午刚面另外一场感觉很累，三哥更累。. visit 1point3acres.com for more.
问了简历，聊了下项目，问了项目里的技术问题，但是年久失修我把B树和DB index的实现原理给记错了。-google 1point3acres
题目是一到系统设计，实现一个leaderboard，第二题是add 2 number。
今天tech talk的时候问了下说是下周出结果通知。



http://www.1point3acres.com/bbs/thread-296343-1-1.html

1. 印度小哥介绍面试流程
2. 简单问了简历上实习期间的项目，是个mean stack project。我先简单介绍了一下，然后他问你是只负责前端，后端，或数据库的设计？然后问为什么选择你写的programming language?问server crash了怎么办？简单回答后就进入到下一步
3. 问了很多java的基础知识 这个时间持续最久 感觉把整个java都问得差不多了 鏉ユ簮涓€浜�.涓夊垎鍦拌鍧�.
4. 2道算法题 第一道算法题做完就已经45分钟过去了，然后面试官说还有一道题 我说好的
第一题：反向打印single linked list。我一开始理解错题目了，写了个reverse single linked list。然后他说不能改变list结构 反向打印list，我问能不能用stack，他估计有点迟疑，因为这个不是他期待的答案，然后他说行，你写吧，然后速速写了下代码；然后面试官让我优化，让我想想怎么样可以用一个loop and without extra space，(面试时脑子真是一片空白，所有答案都是条件反射，反正我就是死都没想到用递归&#128531;)；然后我说是不是可以用stringbuilder，然后反转；然后面试官说跟前面那个一样，然后我尴尬的笑了说对喔，确实紧张，所有的都是条件反射，感觉根本不知道怎么思考，然后他说你再想个一分钟，然后我盯着那个list看，哎，突然觉得可以形成一个integer number啊，以为自己找到答案了 哈哈 然后面试官说不行，如果node的值是string呢，然后我说对哦，然后估计他看我实在想不到，就说，你会递归吗？我说会，然后我才突然意识到原来这么简单
(&#128531;平时很少练习递归，因为感觉听谁说过递归stack有点浪费，能不用就不用，所以我压根没往递归上想，我还顺便感谢了一下面试官哈哈 我说太感谢告诉我这个思路了，我以前很少会往递归上想 哈哈哈哈)，然后让我写了递归代码，写完后；他说嗯，it should work。这时，虽然时间已经45分钟了，面试官说还需要再做一题，我说ok；. 鐗涗汉浜戦泦,涓€浜╀笁鍒嗗湴
第二题：很简单，就是在一个行和列都是递增排序的二维数组里找target，找到返回true，找不到false; 这题简单 秒了 解法也没有争议。从右上角开始，相等返回true，小了，往下移，否则往左移；
.鏈枃鍘熷垱鑷�1point3acres璁哄潧
ps：他们家效率挺高的，第二天早上8点就收到onsite邮件。。。攒人品过onsite...祝大家好运




http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=298177&extra=page%3D19%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

On campus Interview, 问一个和LRU类似的情景，要做一个datastrcuture, 差不多就是同样的逻辑啦～ 可是followup很难， 问一些什么如果数据太多怎么办，或者一个corner case的情况下processing time缩短～




http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=286701&extra=page%3D21%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

电面：
1. Unix系统进程沟通机制，如何pipe多个进程，类似shell是如何实现grep 'test' ./path | head 10 | tail 1，就是unix的一些系统调用，pipe，dup2，fork这些. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷
2. 分配tasks到不同的服务器上。每一个task有cost。我就直接sort，然后用priority_queue做的

Onsite:. 1point 3acres 璁哄潧
1.1 输入是BLMG:100, MS:120, GOGL:134 这样的股票名字和价格，要求设计4个api，1. 插入新数据  2. 获取某个股票的历史最高价格  3. 获取某个股票的当前价格  4.获取当前价格最高的前K个股票的名字； 我用的hash表(unordered_map)，存股票的当前价格和最高价格，然后再维护一个红黑树(map)来获取最高K股票
1.2 C++的知识。读代码，写出代码的输出。主要就是 继承，多态，指针，虚函数表，构造函数的递归调用，析构函数这些+一些unix多进程的知识。这里有个坑是，父类的析构函数没有申明virtual，所以不会激活多态机制

2.1 类似点面1
2.2 unix的程序启动，是一个什么过程，数据如何从硬盘到的内存，swap是怎么回事
2.3 用bash完成 从日志文件中统计 出现频率前五的命令。日志格式类似  time|username|cmd|argv|rst     我也是真不会。。。炸炸炸，然后让我写python的，um。。。python怎么度文件我也忘了，平时python写的少，都是现查doc的。。。就说，let us suppose python can read file like this。。。。。。。
. visit 1point3acres.com for more.
3.1 why bb
3.2 和别人合作的经历
3.3 why this team-google 1point3acres

lunch。。。和三个组员吃，我的心也是在满天飞，哪里吃的下。点菜的时候把12号说成了20号，还被中餐馆的服务生凶了一下，哭
. 鐗涗汉浜戦泦,涓€浜╀笁鍒嗗湴
吃完饭，最后一个大佬还没来公司，team leader就带我去他的工位，给我介绍他的工作流程，演示bb的terminal，一些日程工具什么的，聊了2小时天

4.1 给我介绍了半个小时的bb是做什么，为什么成功
4.2 unix系统程序的内存分布 就是5块的功能

早上9点进去，下午3点出来，太紧张，不停的喝水，不停的跑厕所。。。

分享一点经验的话就是：. 鍥磋鎴戜滑@1point 3 acres
1. why bb一定会问，一定要好好准备，至少准备个不停说2分钟的故事吧
2. why this team 如果你是面的team。一定会很认真的问，每一轮都问了几乎。特别在乎你为什么来这个team
3. 你要有故事。。。你得能扯啊，能聊得来。倒不见得要求英文多好，就是得有话说。我英文特别烂，开口说没事，他们不会纠正你的语法错误的。

感谢地里的朋友发的面经
今天刚刚拿到电话confirm，就来发帖了～
鏉ユ簮涓€浜�.涓夊垎鍦拌鍧�.
最后来一点鸡汤文，我去年12月就毕业了，找了6-7个月的工作。坚持就有希望。说起来今天同时收到了两个offer，直接拒了另一个去bb
祝大家都好运！！！. 鐗涗汉浜戦泦,涓€浜╀笁鍒嗗湴






http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=207184&extra=page%3D21%26filter%3Dsortid%26sortid%3D311%26sortid%3D311




上星期（10/18）面了Bloomberg的onsite。第一轮
一个美国大哥，人很好。
1. 给一堆股票（company, price）， 实时返回当日股票波动最大的k支股票（max value - min value）.. more info on 1point3acres.com
用了HashMap + TreeSet解决的。
2. 出了一道简单题。。 观察数字1231， 3433， 8908， 1091.。。有啥特征。。。其实只是第一位和最后一位一样。
写一个function,检查给定数字是否满足这个条件。
这轮面的不错，小哥走的时候说“希望下一轮的面试官也能给你一样的结果”

第二轮. 鍥磋鎴戜滑@1point 3 acres
印度主面 ＋ 亚裔小哥shadow..
其实有点分不清谁是主面，谁是shadow。。。两个人都很活跃，问了一堆问题。
1. 亚裔小哥
写一个函数，给一个锦标赛制定一个schedule. 比如有A, B, C, D四支队伍。 每支队伍每天只能参加一场比赛， 每两支队伍只能比一次。。
打印schedule如下
Day 1: A vs B, C vs D
Day 2: A vs C, B vs D
Day 3: A vs D, B vs C.鏈枃鍘熷垱鑷�1point3acres璁哄潧
这轮答得不好。。。写出了bug, 忘了去重。。复杂度答得也不太好。。。不过小哥人很好，每抛出一个问题，我都来不及想，就开始试图引领我思考了。。

2. 印度小哥
1. LC125, follow up LC 234
2. LC 214

3. 亚裔小哥
设计一个music player, 讨论shuffle。-google 1point3acres

第三轮
HR姐姐， 等了半个多小时。。。

第四轮
Manager， 人很好，问了简历，behaviour问得很细很细。。。 还问了会看重公司的哪些方面。。
下午2点45左右，walk out.. more info on 1point3acres.com

求大米。。。顺便问一下，BB大概多久会给消息呀，说好的上周来消息呢。。。感觉多半没戏了。



http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=297924&extra=page%3D21%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

上午十一点的面试，interviewer 是个在bloomberg带了五六年的中国小哥， 本科光电（开始5分钟聊到了）人很nice， 面的很认真 鏉ユ簮涓€浜�.涓夊垎鍦拌鍧�.
主要问了internship做的东西，以及课程学了些什么。已经过了25分钟了， 开始coding

1. reverse linked list， 记得写test case；. From 1point 3acres bbs
2. reverse linked list for every k nodes。问这道题的时候已经超过50min的， 只说了思路，我有点着急了接着用iteration了，建议用recursion。





http://www.1point3acres.com/bbs/thread-297865-1-1.html
之前找人内推bloomberg，大概等了一周时间收到邮件on campus面试。
由于是第一次面试，比较紧张发挥不是很好。就把题目跟大家分享一下吧。
给K条LinkedList，每个node节点内部有value和nextNode两个变量，每条List可能无限长，中间可能会有部分list合并，最终K条merge成一条，求出K条分支merge成一条时的第一个节点。. Waral 鍗氬鏈夋洿澶氭枃绔�,
当时lz提到使用hashmap存节点计数，同时扫K条list, 找第一个count为K的节点。但是面试官认为这个方法不是很好，可能需要存很多，占用许多内存。
最后面试官提示说可以在node节点里面加一个内部变量cnt，K条同时扫，每扫到一次该节点cnt加1，当扫到某个节点cnt为K即返回。后来又反复琢磨面试官这个思路，可能LZ见识比较少，总觉得修改node结构这种解法怎么有点迷呢。。。
大家有什么好的想法也可以讨论啊！




上周刚刚店面了bb，一共两道题 lc242和lc139，大约40分钟就做完题，聊了一下职位的相关情况就结束了。
猎头来邮件说还要安排第二轮店面。


BB SDE职位第二次店面，今天的面试官是个白人小哥，竟然没有问coding的问题，一直在问Linux相关的system call，最后问了一些python subprocess和c++ local variable的基础问题。感觉完全不像是码农面试，可能因为小哥是SRE team的。
我准备的是algo coding的问题完全没问，code加起来不到10行。对于那些system call，平时也不常用，回答的磕磕绊绊 :(
最后，发面经求onsite




http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=295957&extra=page%3D23%26filter%3Dsortid%26sortid%3D311%26sortid%3D311
BB环境特别高大上，全透明玻璃和他们的公司tranparency的概念相符，同一时间好多人一块坐沙发上等，有面intern的有全职的有年龄大一些的，估计来面senior。 然后hr到时间了带着大家tour，讲讲历史啥的。
然后领了餐盒就开始面了。除了第一轮，每一轮都问why bb。。。

第一轮：一个国人小哥，人特别好，卡梅毕业的. 鐗涗汉浜戦泦,涓€浜╀笁鍒嗗湴
问了股票那道题和Binray Tree如何返回inorder的list
第二轮： 一个印度人另外一个是印度还是中东 分不清楚
问了一道 linkedlist的题 给定一个array of linkedlist，这些linkedlist somehow 会在中间的某一个地方meet，然后求所有这些lists meet之后 common的部分
第三轮：来了一个美国大哥，在BB工作了15年，是manager了，没有问题，找到lz的一个项目开始问项目，问完项目开始问延展到distributed那些系统问题
第四轮： hr面，问了一些behavior questions，还让介绍一个project，并告知自己不会这个专业的，像讲故事一样讲给她听。问了有没有offer呀，salary expectation啦，BB和其他it公司你选择的时候更看重什么呀。

2点半hr给送出去了。
祝大家offer 多多




http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=297785&extra=page%3D23%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

楼主觉得是第二轮系统设计的 印度哥哥 挂了我.
我在 java 的 treemap 和 treeset 中间犯了一些错误,自己没有理顺清楚,然后他抓住这个点 就疯狂的问 ,然后就挂了.. 鍥磋鎴戜滑@1point 3 acres
第一轮
1) buy and sell  股票 1. 鐣欏鐢宠璁哄潧-涓€浜╀笁鍒嗗湴
2) LFU
第二轮
1) interval merge follow up 2维的 interval  merge 如何做
2) 设计个 report 系统. 这个系统对最近的股价 log 从最高值 到最低值 输出.
如果股价有修改, 删除旧的 输出新的.


恩, 应该是在第二轮上出了问题. 然后被请出来了.






//http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=289661&extra=page%3D23%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

TOMS第一轮 国人大哥+大姐
国人大哥
is string palindrome.1point3acres缃�
follow up1: palindrome partitioning（被要求暴力扫）. From 1point 3acres bbs
follow up2: 不管palindrome，递归中可能的所有组合。(eg: "1210" -> [1,2,1,0], [1,2,10] ..) 要按顺序，也就是演示一遍代码. From 1point 3acres bbs
follow up3: 时间复杂度，为什么是这个时间复杂度
国人大姐
LFU
follow up： 感觉是自己挖的坑导致的follow up。 我就问了一句如果删除次数低的是考虑相同次数中先到的，还是任意一个，被大姐反问，你觉得呢。我就说了两个的差别，然后引出了linkedhashset。大姐就说我不懂linkedhashset是什么。然后我就解释说是java里面自带的一个，有queue和hashset的特性，大姐还是说不懂，我就只能说如果是我要自己实现这个linkedhashset要怎么写，因为我觉得我也解释不清楚。然后我就开始了如何实现linkedhashset，hashmap+linkedlist，大姐说你觉得你要用linkedlist还是doublelinkedlist。大姐你还说你不懂。然后我就大概解释了一下我要怎么做怎么做。大姐最后说了句，恩我懂了。套路啊。。。。

第二轮 国人小哥*2
小哥1：设计
vending machine，里面有零钱，有产品，要求实现1.display()//显示产品信息， 2.addcoin//塞钱， 3.deliverProductAndReturnChange//給东西， 4.cancel, 5. reset。小哥整个说话都是颤音，这不是应该我紧张吗？然后我写出来了。
followup 1: 如果现在我产品分food和drink, vending machine 只能是foodvedingmachine 和drinkvedingmachine. 要怎么改我现有设计. 我就说继承vendingmachine咯。但是我一直没get到点，另一个小哥就点了一下，原来不是继承vendingmachine, 是food 和 drink 继承product。
followup 2: 多种付款方式
小哥2：Database(可能小哥的本意不是这个，我又给自己挖了个坑）
小哥让我解释了一个3年前的project，还好我之前有回忆一下。小哥问的是我项目组db相关的东西，然后就给我讲了一个表，里面有什么东西要做一个report，就是类似于group by, 在他讲的过程中我一直就在想着sql要怎么写，然后小哥画风一转说那如果我们要设计一个datastore。我一听不是写sql，一忘形就冒了一句我以为你要让我写sql呢。小哥一听，可能觉得有点意思，那就说我们来写sql吧我又把自己坑了，问题是我也没想出这个sql怎么写。然后我就写group by.
follow up1: 小哥说这个耗费有点大，价格filter吧，难道是having？ 我不是很明白，后来才知道小哥只要部分，并不是要全部数据。加了个having
follow up2: 怎么改可以加快查找速度？ 加index。小哥又问是给所有column都加吗？额，难道不是？小哥看我蒙圈中，就又补了一句，还是只给那些需要查询的加。
. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷
第三轮 白人大哥 + 不知哪国大哥
我当时觉得我不会有下面的了，毕竟前面两轮我各种给自己挖坑。等了一会，又来了两个人。
他们也没有说他们是manager，一来就给我讲他们组干什么的，然后我一直处于郁闷中，也没怎么认真听。他们分别讲完了过后，就说你有什么要问的，凭着听到的关键字就问了一些。
然后就是常规 why bb？ one project? 等等
. 鍥磋鎴戜滑@1point 3 acres
lunch

第四轮 白人大哥*2
听带我吃饭的那个小哥说，其中有一个是这个组最大的头头.1point3acres缃�
why bb？ one project？  如果我是manager招人最看重的点还有最不喜欢的点
. Waral 鍗氬鏈夋洿澶氭枃绔�,
DASH组
第一轮 白人小哥 + 不知道哪国小哥.1point3acres缃�
白人小哥题忘了，但是用hashmap. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷
另一个小哥，类似股票那个题，update, top10，不过分了buy和sell，我就直接写了个，另外一个逻辑相同

第二轮 不知哪国大叔 + 天竺姐
被大叔套路了一下，问了我一个项目，然后问为啥想换工作，我就模板回答啊没有挑战啊balabala，大叔就接着问为啥没有挑战，我又模板回答，然后大叔就问，那你说说你不喜欢你现在公司的点，我赶紧说我没说我不喜欢啊，然后又解释了一下，大叔就给了一个结论，你就是说现在工作做得有点无聊吧。这是什么套路啊，我感觉接着扯没有这回事。然后大叔在那得意的笑了，无聊就无聊嘛，这是正常的啊。然后让我问他，我就问你既然问了我不喜欢现在公司的点，我就问问你喜欢b家的点。大叔还专门强调了一下你问喜欢还是不喜欢，我说喜欢，我又不傻问不喜欢干嘛。

天主姐上题： 鏉ユ簮涓€浜�.涓夊垎鍦拌鍧�.
word-page那个题。要求是只能用最基本的数据结构，就是不能用hashmap啊
我就用trie-tree写, addword(word, page), findpage(word) 鏉ユ簮涓€浜�.涓夊垎鍦拌鍧�.
其中我想用java自带的linkedlist存page，大叔说不能用，好吧我又默默的自己写个linked list
followup: 时间复杂度，如何提高查找速度，这个时间就可以说我要用什么hashmap，hashset之类的东西了.鐣欏璁哄潧-涓€浜�-涓夊垎鍦�

第三轮 白人大姐 manager. From 1point 3acres bbs
让我讲了一个项目，然后给我演示了一下terminal，讲她的组做了哪些部分。
我想都已经快5点了，可能是最后一轮了，我就用尽我的洪荒之力在那惊叹terminal 好厉害，我本身也是觉得比较厉害，但是有过度表演的嫌疑。我明显看到manager的脸被我夸红了。然后她就让我问她问题，我还在不停地夸这个太强大了，你们组复杂的哪部分啊。

第四轮 白人大叔 manager
都5点多快6点了，一个白人大叔又走进来，说他是manager，然后还说我知道你已经被关在这个房间里面一天了。我当时心中就只有我的个天还有，但是我居然条件反射回了一句，没有，我上午被关在另外一个房间里面。. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷
然后大叔就直接开始画图，画了两个房子，一个车，一条路，我一脸懵逼的看着这是要干嘛。大叔完成他的图之后，就在说我从我家开车到我老婆娘家要走几段路，距离啊速度啊，然后我有个GPS啊，它就会告诉我开过去要多长时间。这是什么套路，然后大叔继续画，说你来设计这个GPS，如果途中堵车，然后GPS该怎么计算。我在上轮已经用尽全力了，现在脑子完全转不动，然后就在那想google map是怎么搞的呢。然后大叔又问我堵着堵着发现原来是出车祸了，然后我绕过这个路，接下来gps又怎么给我要剩下多长时间。然后我就说你绕过这段，你也不能一下就到之前的速度，要慢慢的加，然后我一直想不起来加速度怎么用的。我就取首尾的速度一段一段的来凑。然后大叔就说我可能这段距离中速度可能是 ^ 这样，你这样就不行，我就是那取平均吧。然后大叔又问我开累了，我跑到路边去加个油喝个咖啡，然后重新启动gps，这个时候gps怎么算剩余时间，我真的已经想不出来什么了，就想起我的车每次计算mpg怎么搞的，我就说看你停了好久，如果时间不长，那就继续之前的计算，如果时间长了那就reset。大叔就问，那你觉得这个window要多大，我就说10-30min吧。然后大叔又说，在节日的时候有惯性堵车，那gps又该怎么算，我就说取去年的记录来看看。然后大叔还问了一个什么给用户最好的预估还是最坏的预估。我当时已经舌头开始打结了。就说如果gps的话，要求3点到，但是我会计算出2点50到的话的开始时间让他开始，免得中间出意外情况。
大叔话锋一转，你不要认为我们是做gps的哦，我就说感觉你的描述有点想股票价格的曲线之类的。然后就开始给我讲也不是价格，是其他东西，然后balabala给我科普了一下finance的知识啊。我已经处于看着他嘴巴在动，就只能时不时听进去几个单词的状态中了。然后大叔又让我问他问题。我就是一个很小的问题，他可以讲好久好久好久好久啊。。。已经6点过了，大叔就说hr一直坐在隔壁会议室，我等会给她说一下让她问快点。.1point3acres缃�
. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷
最后大叔跑到隔壁房间，跟hr在那聊天，我就等啊等啊等的，10分钟左右吧，hr过来就说今天太晚了，送我去机场的车在楼下等着的，明天给我打电话，然后把我送下楼。然而就在刚刚，hr给我打电话我接起来然后听不到声音，她就给我挂了，现在再说重新约时间。

感觉见了好多人，从早上9点到b家，到下午6点多才出来，面组里面的大boss总是在状态不好的情况下，吃完饭犯困中和洪荒之力用尽之时。希望自己不要被套路了。发个面筋求offer。
感觉里面的人都蛮激动自己在做的东西，里面也有我自己很感兴趣的一些东西。虽然听说扭腰很冷，但感觉是跟LA完全不一样。





http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=293357&extra=page%3D23%26filter%3Dsortid%26sortid%3D311%26sortid%3D311
内推的Bloomberg 本来on campus但是没时间改店面了印度小哥感觉还比较nice
上来先问了why bb和简历两个project 然而大兄弟拿的我的之前的简历，project我都快忘了...
coding第一题inplace merge two sorted array
第二题inplace remove space in string
都是two pointer




http://www.1point3acres.com/bbs/thread-297780-1-1.html

一开始聊最rewarding的项目, 然后就是做题. 类似里口二三就, 只是要求所有窗口最小值中的最大值. 首先brute force, 讨论了一下特殊情况, 也就是窗口大小是负数怎么做异常处理, 就说了抛出异常或者定义一个表示invalid的constant返回.
.1point3acres缃�
然后followup就是缩小时间复杂度, 用的dequeue的方法, 就是在窗口里维护数的index, 而这些index上的数从最后到最前是递增的, 可是当时脑抽, 不知道哪里弄错了, 跑的test case不对. 面试官说我知道你的意思了. 然后就问问题了.

还以为挂掉了, 没想到给过了. 这是面bloomberg第二个组的第二个店面, 第一个组面完了二面收到HR电话问了几个HR question说过几天会通知下一步.....这次二面结束后过了20分钟, recruiter就说过了,想约个时间聊一聊






http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=295050&extra=page%3D27%26filter%3Dsortid%26sortid%3D311%26sortid%3D311
分享一波彭博电面的面经。
面试官：中国小姐姐；
Behavior Questions: 自我介绍之后，小小问了一下简历之前的项目经历， 一句话回复之后准备细讲 结果小姐姐说不用了
Code:
1.一道题好朋友谁比谁高(1,2)(2,3)(3,4)(5,6) 1比2高，2比3高，问你1 和 3谁高，有三种结果 ，高，矮和不知道。
2.利口  思思吴原题。



//http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=297219&extra=page%3D27%26filter%3Dsortid%26sortid%3D311%26sortid%3D311
一个很nice的东亚小哥，听不出是中国韩国还是日本等等。。在bloomberg多年
1. 找到出现连续出现次数最多的字母，如果次数一样随便return一个
"ccaaaccbbdbca" => 'a'

2. 先升序后降序的array找到一个target，如果没有return -1，有的话return index
[11, 33, 675, 1999, 98, 0, -3] 找 33




//http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=296379&extra=page%3D27%26filter%3Dsortid%26sortid%3D311%26sortid%3D311
上来介绍了面试流程，先了解背景，做一道两道题，最后question to ask
自我介绍完本来期待会问why bb，结果没问就直接做题了
第一题fizzbuzz，感觉可以直接秒
第二题用stack 实现 queue，也一遍bug free
一看时间才过了十五分钟。。。
他说下一题本来不是给java面试者准备的，是c++写的一个struct 但是我可以用java实现.鏈枃鍘熷垱鑷�1point3acres璁哄潧
一个node有上，下，右 pointer 然后便利
1
|
2-google 1point3acres
|
3->6  比如是这样的就要打印 1 2 3 4 5 6.. From 1point 3acres bbs
|
4
|
5

也很快做完了，，，问问题的时候本来准备了两个问题，后来时间太多就多问了一个又聊了一会。。。. visit 1point3acres.com for more.
不知道会不会有结果，，求onsite。。上一次微软校园面本来以为挂了，发了面经就有了onsite，希望这次也有







//http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=297391&extra=page%3D27%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

自我介绍
Validate 二分ST
decode string变成二分&#127794;
面的不是很好，但是基本思路都说了，当初出来感觉挂了





. more info on 1point3acres.com
然后，一小时后hr约onsite



补充内容 (2017-10-11 22:27):
decode string变成二分树








http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=294026&extra=page%3D27%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

回报地里并求大米攒人品
电面1：类似于leetcode里的“guess number",给你一个算投资回报的公式，让你用已知变量求利率，（利率是一个指数所以不能通过数学方法转化公式来求）.鐣欏璁哄潧-涓€浜�-涓夊垎鍦�
就是用binary search 来做，只要注意这是一个continuous range, 所以要设一个最小误差

电面2：
就是leetcode里的 "course schedule II"， 只不过把course number 换成 job id

以上两个是structured product 的组. 1point 3acres 璁哄潧


还有一个MARS 的组电面：
leetcode 332的变形，没有方向， 起始和目的地是给定的参数。 用DFS 做，注意在dfs 之前要先把起始位置标记好，并放到result 里面，我就是在这里出了错






http://www.1point3acres.com/bbs/thread-297220-1-1.html


双十日电面：人不错的三哥1. 以前的经历
2. 利口二时
3. 利口三菱幺

应该fail了。。。



http://www.1point3acres.com/bbs/thread-295105-1-1.html

 聊简历, 最近的project, 有没有要提高的地方, 最难得地方是什么(这年头, 店面也问几个BQ)
- 询问java的Collection, LZ提到了Map, 于是顺势就被问了HashMap 和 TreeMap的区别, 就时间复杂度和空间复杂度说了一下.. from: 1point3acres.com/bbs
- 设计一个keyvaluestore, 但是不要用Map, 用ArrayList, 来做, 问时间复杂度, 回答O(n), 又问可不可以加快query()的, 就说维护一个排好序的list, 然后用二分法查找
- 算法, 把一个数组里面的数根据出现的frequency排序, 相同frequency的值小的排在前面, 比如 8, 2, 2, 4, 9, 9 => 4, 8, 2, 2, 9, 9
- 问两个进程如何交流, 两个进程怎么同步

45分钟结束, 最近面的快麻木了, 马上10月了, 还是要接着面, 接着刷题准备.






//http://www.1point3acres.com/bbs/thread-296022-1-1.html

发了面经，攒攒人品吧。
第一轮：一男一女白人面试官，都很友善。第一题的情景是给input公司A是公司B的母公司，公司B是公司C的母公司，然后写个method判断公司A是否是公司B的母公司，有向图解决，然后优化是直接从子公司开始找，因为可以assume每个公司最多只有一个母公司。还有一题是LC留时尔变种，每个方格有数值，找path sum最小。中间还有一题实在想不起来了，不过也很简单。结束约了下午第二轮。
鏉ユ簮涓€浜�.涓夊垎鍦拌鍧�.
第二轮：一男一女面试官，也挺友善的。第一题就是考选择什么data structure现在想想其实就是hit counter。。。不过他还要求你能够找到前1小时内top k的record的什么的。然后男面试官follow up,如果可以牺牲accuracy,可以怎么减少内存，后来面了dropbox发现和dropbox hit counter的follow up很像，可以用那个什么circular array。基本思路就是比如这一秒hit的数记在一个array的element里面，然后抛弃的时候也是1s内的都抛弃，这样就不用用queue存所有记录了，但是会牺牲accuracy。面试的时候弄了很久才明白他们的意思。。。最后也差不多设计出来了。然后code考了一道LC妖妖漆。结束约了几天后的一轮。

第三轮：HR。Why Bloomberg？Why software engineer? 用非技术语言解释一个项目。

第四轮：白人Manager。大部分时间是他在讲，介绍他在BB的工作什么的，然后问了问简历，问了问经过了两轮面试，对BB的感受。最后展示了terminal估计展示了得有二十分钟吧。

整体体验挺好的，说是一周后给结果。不过他家好像四轮之后还是刷很多人，所以随缘啦。希望招工季顺利一点。







http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=292844&extra=page%3D31%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

第一轮
第一题本质就是dfs bfs 但是题目太花了没想到 最后写了递归不会算递归的复杂度还是面试官教我的 （转专业的痛） 感觉这轮凉了啊
第二题LRU类似 讲了下用什么数据结构和几个主要是操作步骤 corner case

第二轮
第一题 uniquepath 一开始写了个O(MN)空间复杂度的DP解法装作没见过等着优化 结果两个面试官都没见过这个解法 解释了半天 我说可以优化空间复杂度也没要我优化 面试官说可以用recursion解法时间复杂度不用O(MN) 不过我写的能行就下一题了 有大神知道怎么recursion做到时间复杂度小于O(MN)吗. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷
第二题 “你安排你的朋友下个月一起旅行的时间”
就一句题干 剩下自由发挥 就每个人的时间都用queue就行
第三题 一排小孩报数，每n个出局，求最后剩下的. visit 1point3acres.com for more.
一开始用array记录出局的人 每次数n个非常麻烦
面试官说你熟悉linkedlist吗 秒懂画了个doublelinkedlist 说首尾相连 idea差不多了就不用写了
第四题
问还想做吗 就再出一题吧 本质就是移0 讲了idea面试官说做过吧我说做过 然后就问问题了

第一轮面的不好 他们家不怎么要求bugfree 求rp



补充内容 (2017-9-17 06:33):
hr通知下周终面 依旧on campus 只剩一轮了 求rp

补充内容 (2017-10-10 06:34):
offer了 orz 实在等的心累了



http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=293598&extra=page%3D31%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

校招第三轮，昨天下午面最终面，hr+疑似manager的亚洲姐姐（怀疑是国人 听不出口音）
没有算法题，聊了 很久的简历和实习project 问了些bq
技术问题出了个 大概就是很多数据输入 同时很多终端不同客户要显示的输出是自己定义的 问怎么设计。。
复述不出来因为没学过 我就把所有我看过的面经里的答案讲了一遍 不知道讲的咋样（真的没学过）
hr问我最近是不是有很多面试 有没有preference
我说没有preference我比较喜欢nyc和seattle（其实一点都不喜欢nyc。。。）（现在真的后悔 应该说最想去bb的 非常喜欢金融什么的...)
感觉表现的不是特别特别想去 说是一周到两周出结果 前面两轮的feedback还没收到
感觉一面面的不好二面还可以 三面技术题感觉面试官的反应还行虽然我不知道正确答案是啥都是胡扯
求offer 顺便为明天后天的onsite还有下周的四个面试攒人品T.T



http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=296985&extra=page%3D31%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

今天又面了另一个职位，面试官说话很快，还好没有口音。简单介绍了bb之后又问了whybb的问题。
然后进入正题，LC器灵，dp做完之后followup是一次可以上1-N阶台阶。求总数。当时想了想可以把 i-1 ~ i-N的都加起来，但是写的时候磕磕绊绊，最后在面试官的提醒下才写完运行通过。还问了time和space complexity.
发面经求onsite



http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=296965&extra=page%3D31%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

猎头说有三道题，但是结果没时间只做了两道，应该没戏唱了。
第一道题 le 451变种， 与其重复越多的摆在最前面，现在重复越多摆在越后面。
第二道题 写一个程序， 按照class make file 的正确顺序存进一个 vector。例如:
class A: B, C
class B: D
class C.鏈枃鍘熷垱鑷�1point3acres璁哄潧
class D

正确顺序是D, B, C, A
method arguments 随意自定， 我用了unordered_map。
-google 1point3acres
之后就让我问了下他们团队在干嘛，每天的工作性质是咋样的等等




一个美国小哥，看见我专业是EE，狂问数据结构，LinkedList, ArrayList, HashMap, Tree, Stack, Heap问个遍，楼主英语略渣，虽然感觉很简单，但答得不是很好
coding，就一道题，给一个数组，找众数，用hashmap 写了一遍

然后优化一下代码，小哥说coding就到这里
. 1point3acres.com/bbs
问了一下如果不用其他数据结构怎么做，我想了想说sort一下. 鍥磋鎴戜滑@1point 3 acres

然后就问问题了。。。 才33分钟

感觉小哥人挺好，但是口语实在太渣，感觉GG了



一个美国妹子面的， 题目超级简单，merge two sorted array. Follow up: merge million of sort array(either ascending or descending)。最后问了下时间复杂度结束了。美国妹子放水很严重。。。。。。




刚面完BB。超级新鲜面经。
1. lc 445. 我用Stack做的。问能不能用O（1）space，想了半天没懂，他说先reverse一下就好了。丧脸。
2. lc 33. 太紧张了写了好半天才写对，草稿纸一定要备好！不然脑子里画不出图来！
反正都是lc原题，我太紧张了。求人品！！！







http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=296006&extra=page%3D33%26filter%3Dsortid%26sortid%3D311%26sortid%3D311


发帖回馈一下(?)社会，据不可靠观察发现BB家好像一直在我们学校招到十月底，希望这个帖子对后人还会有帮助:)
BB一共面了3+0.5轮，在校园招聘会上直接给Engineer递的简历，当时被问了很多问题，然后聊的挺好的，当晚就收到了第二天的on campus interview
接着就是连着三天的on campus，题目都不难，但是很累人
第一轮 tech 两个面试官 问了一些关于简历的问题和3个coding problem
1) Given a book, find the five most frequent words
2) Partially reverse a string e.g.: "Fruit flys like bananas" -> "tiurF sylf ekil sananab". From 1point 3acres bbs
3) Count the maximum distande between nodes at the same level of a tree
面完第一轮直接就告诉我进第二轮了 然后约了第二天
第二轮 tech 依旧两个面试官 问了一些关于简历的问题和3个coding problem
1) Customer知道一个数字，你要准确猜出数字，你只能给Customer一个数字，他只能回答T/F，你可以随便问几次
2) Number of islands 这个算是BB经典题了吧
3) 忘了……
面完直接schedule第三轮
第三轮 manager
其实是最方的，因为听说有个学长面了两年BB，都是在第三轮完了之后再也没有听到任何消息……
一个印度manager，非常不给shit，你说什么他都非常冷漠hhh但是还是努力找话题尬聊，为啥要来BB，你之前的实习经历blablabla
我觉得比较关键的是当他问你有没有在跟别的地方interview的时候你一定要斩钉截铁的说NO
第三点五轮 HR
聊啊聊啊聊
然后过了一周顺利地得到了Offer
完
有问题可以在帖子下面回复:3
.1point3acres缃�
. 鐣欏鐢宠璁哄潧-涓€浜╀笁鍒嗗湴
补充内容 (2017-10-6 01:15):
二面第一题还有另一个function可以take in一个数，然后告诉你target number是不是比这个数小






http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=293234&extra=page%3D33%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

BB 两轮电面
都是白人大哥，每轮都有background 和proj 介绍
第一轮：find the kth smallest element in an unsorted array.. 鐣欏鐢宠璁哄潧-涓€浜╀笁鍒嗗湴
第二轮：（大哥特别冷漠，好几次都以为他掉线了）。。。. from: 1point3acres.com/bbs
1. 设计数据结构，存取删除都是O(1)，然后还有按insert顺序iterator功能，问存储空间. 鍥磋鎴戜滑@1point 3 acres
2. 给个string，找subset，用recursion不满意，要用bit做，要跑test case


补充内容 (2017-9-26 19:24):
都答出来了，还是fail了，move on，BB题目不难，朋友们好好准备


http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=296183&extra=page%3D35%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

两轮四道题:第一轮， 白人+国人姐姐
1. Josephus problem. 鍥磋鎴戜滑@1point 3 acres
2. LC buy and sell, 交易次数没有限制，输出最大获利的各个买卖点
第二轮， 印度人+国人姐姐
1. BST to double linked list
2. two sum
. Waral 鍗氬鏈夋洿澶氭枃绔�,
挂在BST to double linked list那道上了。








http://www.1point3acres.com/bbs/thread-296293-1-1.html

面试的时候竟然codepair当掉了，期间尬聊五分钟，codepair回复正常
一道题
存最近的N个不同电影观看记录
e.g. (N = 4). more info on 1point3acres.com
当前： A, B, C, D
看了E后：E, A, B, C
看了B后：B, E, A, C
让设计一种数据结构实现上述功能




http://www.1point3acres.com/bbs/thread-283568-1-1.html

因为卤煮不在纽约地区，被要求电面了第二轮上一轮面经： http://www.1point3acres.com/bbs/ ... adio%26sortid%3D311. visit 1point3acres.com for more.

这次是一个口音很重的小哥面试，感觉像法国人或者意大利人讲英语...于是卤煮出现了沟通困难综合征...
面试总共就一道design题（是的运气也是好连着两轮design题电面）

上题：
设计一个MTA Garage system,  给你每台巴士的schedule的文本(bus id, entry time, exit time)，然后让你design一个system，支持getCount(timestamp)，该函数返回当前在garage里面巴士的数量...
小哥不让用binary search，而且纠结了好久我用vector<string>作为输入的问题... 发面经求人品求昂赛








http://www.1point3acres.com/bbs/thread-162025-1-1.html

一轮电面：
白人小哥口音，上来聊了聊简历，问做过的最有趣的一个project。然后出了一个题，一个类似facebook的社交网络，对于用户A, 写一个function计算网络中其他节点相对A的得分，他描述题目就描述了很久，大致是类似pagerank如果B是A的朋友，或者是A有很多朋友都认识B，那么B的得分较高，依次类推。这题做完之后就没时间了，估计面试官心中也没有完美的解法。。。。

Onsite:
第一轮：中国小哥+白人大叔. 鍥磋鎴戜滑@1point 3 acres
白人大叔上来问了一个hashmap的get操作如何优化其worstcase的时间复杂度，然后又问为什么标准库不用这种优化方式。
第二个题是中国小哥问的 validBST， leetcode原题。 鏉ユ簮涓€浜�.涓夊垎鍦拌鍧�.
第三个题是白人大叔写了一串代码，然后问了很多java string的细节，包括immutable，还有内存的allocate等等。

第二轮：亚裔小哥+印度小哥
第一个题，给一个手机键盘和一个字典，2对应abc，3对应def。。。。等等， 要求写一个函数，输入是“223”这种数字组成的字符串，输出是所有这种字符串能产生的字母组合中在字典里出现的字符串。
很简单的backtracking，follow up 是如果可以对字典进行预处理，如何用O（1）的时间得到数字串对应的字符串列表。
第二个题，要求写一个函数，计算两个正整数的乘积，不允许用*，/，%，只允许用位运算和加减。要求时间复杂度less than O（n）。
第三个题，让你设计一个list容器，然后可以支持整数类型和null的存储，我的解法是再开一个list来存null的index，后来经提示用bitvector来存null的位置。. 鐣欏鐢宠璁哄潧-涓€浜╀笁鍒嗗湴

第三轮：亚裔manager
聊了一个most challenge program，聊了一些behavior question 还有 why bloomberg之类的。。

第四轮：白人妹子recruiter. 鍥磋鎴戜滑@1point 3 acres
聊了一下选择offer的因素，同样 why bloomberg，然后说入职会有12周的培训，之后match group之类的。。。。
-google 1point3acres
早上11点正式开始面，中间基本每轮完了有10分钟，下午3点走出bloomberg大楼




http://www.1point3acres.com/bbs/thread-296098-1-1.html
听不出来小哥是哪里人
为什么bb
实习的问题 各种问
小岛数
如果不能modify input 怎么办







http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=294463&extra=page%3D37%26filter%3Dsortid%26sortid%3D311%26sortid%3D311


PHONE SCREE 非常简单就不提了 基本的两个ARRARY数字相加，C++语法问题以及一个 TICKER MSG DESIGN 问题。. Waral 鍗氬鏈夋洿澶氭枃绔�,
on site:
第一轮， 年轻老中+年轻烙印.
1) 字符 )(())(hello)(( , 如何 in place 把无效的括号去掉. .鐣欏璁哄潧-涓€浜�-涓夊垎鍦�
2) a<-> (b,c,d)
    b<->(a,d,e)
假设有一个MAP INPUT，KEY是 一个NODE，  value 是这个NODE相连的其他NODE的LIST，
如何找出从A 到 D的所有路径.

2)老中和烙印.
vending machine OOD.
design一个 order msg server,如何REAL TIME发送到client,以及维护没有成功发送的. msg

3) TEAM LEAD 老中和烙印.
问的非常杂，从最基本的 TREE问到红黑树，复杂度，推算复杂度，各种排序比较，优化。
如何设计一个一秒内被call 1000 次就报错的 func,用什么数据结构最快.

4） 午饭，很友好的一个老中和一个烙印。
午饭期问了问之前的经验，SQL和UNIX熟悉不熟悉之类。

5） 大老板，
各种BEHAVIOR问题，表忠心， 大老板说 we will def get back to you next week.
6） HR。
各种BEHAVIOR问题， 要多少钱，什么时候能加入云云.

被送走，从早上9点一直到下午3：30. . visit 1point3acres.com for more.

周五面的， 周一晚上6点半收到据信。

不知道那一轮被竖了 reg flag,LZ 表达能力还可以，毕竟本科开始在职工作了很久， 所以交流很流畅， 但算法和年轻人不能比，不是科班出身，写的不是非常流利。但基本都磕磕碰碰给出了解法 。.鐣欏璁哄潧-涓€浜�-涓夊垎鍦�
OOD那道答的不好， 面的非常着急，没来得及看，大家要小心。

不知道具体栽在了那一轮.可能LZ在面大大老板的时候表现的张扬了一点，大老板是个挺年轻的印度人，感觉深藏不露，
很有意思的是组里一半中国人一半印度人，不知道会不会因为LZ表现欲比较强烈英文比较流利引起了印度大老板警惕被干掉了？






http://www.1point3acres.com/bbs/thread-295570-1-1.html

第一题是堆盒子，给一个列表的盒子，盒子有长宽高三个属性，只有下一个盒子长宽都小于stack顶端的盒子的长宽才能被堆上，求可能的最大高度
第二题是字符串括号匹配那道，加了单引号匹配



http://www.1point3acres.com/bbs/thread-294443-1-1.html

第一轮：国人姐姐和美国小哥，国人姐姐shadow。第一题判断一个linkedlist Palindrome, LC234吧。. 1point 3acres 璁哄潧
第二题，给我一堆列表，分别是子公司和母公司关系。写两个方法，一个是拿到一个给定公司的直接母公司。另外一个是判断给的两个公司是不是子母公司关系。
例子[(c1, c2), (c2, c3)]， 方法1:给c3， 返回c2；方法2:给c1和c3，返回true。

第二轮：国人大哥和印度哥。
一道题，写两个api。大体意思是decoding和encoding。
encoding就是把一个string数组合并成一个string（格式自定），decoding就是把这个string解析回string数组。. 1point 3acres 璁哄潧
follow up是如果这个数组里面的元素长度大于10如何解决。
然后问了一些system design的题，问的云里雾里答的云里雾里，估计挂在这里了。



http://www.1point3acres.com/bbs/thread-293939-1-1.html

有两题, 都很简单
1. aaabbbccc -> a3b3c3
    aabbcc -> aabbcc
    字母连续3个一样要变成数字

2. 给一个N长度的 ARRAY 里面的数值一定在 N 以內. 鐣欏鐢宠璁哄潧-涓€浜╀笁鍒嗗湴
   比如说 N是5, 那ARRAY可能是[3,1,1,4,0], 找重复的
   给了几个O（N）SPACE COMPLEXITY
   好像不太接受
   最后给了一个O（N LOG N）TIME，O（1）SPACE好像可以了.1




http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=296027&extra=page%3D39%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

   1. 给出parent company，child company pairs，设计data structure要求1）给出一个child company，可以很快找到它的parent company。2）给出一对company a，b，判断a是不是b的ancestor。很简单hashmap。
2. 一个6位数，如果前三位的和与后三位的和相等，则是good number。问所有的6位数里有多少这样的数. 1point 3acres 璁哄潧
3. 从1开始，只能*2或者/3。给出任意一个数问要最少几步运算可以拿到。




http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=295897&extra=page%3D39%26filter%3Dsortid%26sortid%3D311%26sortid%3D311
博三学生，找暑假实习，career fair上投了bloomberg，不到一周（这周一）收到面试通知，本来可选时间是周二、周三，但是周三被秒抢。。。不到24小时准备
两个面试官，上来先随便聊了聊简历还有为啥选择投bloomberg，之后上technical问题
1) 输出两个char array，bank和line，要求输出是否能用bank里面的字符代表line
我上来用hashmap，经过提醒用两个pointer一起扫这两数组更快
2) 一个链表，判断是否是palindrome.鐣欏璁哄潧-涓€浜�-涓夊垎鍦�
我上来用stack，后来问能不能降低空间复杂度，我以为要用constant呢，没想出来，后来提醒了一下发现是找链表中点，翻转链表
3) 一个图，图上每个节点里面一个char，先问怎么表示图，我说邻接表，让我写数据结构，之后问我就说每个节点作为root做dfs，考虑了一下怎么存visited node的问题，后来觉得hashset靠谱就用了hashset

不知道为啥考了我3道题，我算法知识还是欠缺，虽然都能解决问题，但是一旦问优化就不行了，感觉都不难，没要求写太多代码，主要都是说思路

基本跪了，总结一下造福诸位吧


补充内容 (2017-10-4 22:45):
第三个问题要求写出所有能被这个图中的connected node表示的string



http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=295851&extra=page%3D39%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

9月份网上海投的，可能自己简历上做的东西跟他们家还蛮match的吧，又或者是为了增加diversity。。
。上周通知本周on campus 面试，1h，一个貌似是国人的大哥，一个白人大哥，都是工作了5年的码工。首先介绍了各自的team的工作内容，
然后就是国人大哥问了一道 利口斯旧流，之前没刷过，这里卡住了很长时间，国人大哥一直在给提示，后来给出了一种linkedlist的解法，
大哥说这是他至今为止看到的第二人这么做，很有趣，感觉为了给我加戏真是辛苦了；白人小哥问了下stack的实现，估计是因为第一题没用stack做所以才问的。
代码都是在纸上写的，不过也可以写白板。最后5min，就随便问了他们几个问题。总之还是跪了，但是楼主真的蛮喜欢他们家做的东西的，move on了==




http://www.1point3acres.com/bbs/thread-295927-1-1.html
蓬勃今天校招
两轮都没问我why bb
服了。。。昨天准备了一晚上。。。. from: 1point3acres.com/bbs
第一轮：find the k most frequent element in an array 讨论了半天follow up和时间复杂度
第二轮：我觉得最神奇的地方是bb这么大题库俩题差不多长一样？？？用bucket sort做了一下，第二题是 lc 壹叁扒.鐣欏璁哄潧-涓€浜�-涓夊垎鍦�
面试官人都超nice
第一轮俩印度哥哥一直在跟我开玩笑一直给提示然后我就特别开心和放松。。。. From 1point 3acres bbs
求下一轮


补充内容 (2017-10-7 02:43):
今天收到了拒信 不move on了。。。
第二轮感觉答的也可以 两道题代码很快写完也说清楚了思路。. from: 1point3acres.com/bbs
唯一觉得答的没有那么好的地方可能是一开始问我一个system design的题我没什么思路这样吧。
真的很喜欢BB&#128546;




//http://www.1point3acres.com/bbs/thread-293902-1-1.html


9/20参加的bloomberg on campus第一轮，时长一个小时。一对二，一个白人小哥哥，一个印度小哥哥。现在大概问了问简历上的东西，然后开始做题。白人小哥哥出了道lc451，加了一个条件，freq底下还得按alphabetical order输出。写完代码问了问复杂度以及怎么优化tradeoff之类的。印度小哥哥问了一道tree的题，没见过原题但思路类似zigzag那道，说我不是最优化不过也没时间了...在我之前一天面的收到了第二天的第二轮，但到我这他说...预定的会议室全满了，大概等一到两周hr通知，瞬间就有些虚...希望好运吧，也祝大家收获offer~
. Waral 鍗氬鏈夋洿澶氭枃绔�,

补充内容 (2017-10-19 23:15):
10/19二面，一个小时。依旧一对二，聊完简历两道题。第一道给一个2d vector，每一行sorted，让你输出一个sorted array。第二题，设计一个lottery机制，可以加名字进去，删名字，和随机获得一个得奖者名字。







http://www.1point3acres.com/bbs/thread-292311-1-1.html
上来先让我自我介绍一下, 然后问了, why BB...

Tech:
1. LC 20, 很简单的一道题.
. more info on 1point3acres.com
2. follow up,
以下内容需要积分高于 133 才可浏览

问如果用户想设定matched pairs, 应该怎么做,
    比如 想设置,  只关心,  '/' 和 '\' match,  '^' 和 '*' match;  这里假设matched pairs are one-to-one and unique,  比如不存在 '/' 和 '\' match, 并且 '/' 也和 '|' match.. 鍥磋鎴戜滑@1point 3 acres
    让我自己设计个函数, 规定输入的参数.
     挺简单的, 我就做了个map 参数,  比如  boolean isValid(String input, Map<Character, Character> map),  用map来做检查
. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷


3. 问如何从一个 integer 数组中找出3个元素的最大和,  应该也是LC上的某道题 (一时找不到题号了....)
    就让我说了下大概思路,  其实就是 Math.max( max1*max2*max3, max1*min1*min2 ).. From 1point 3acres bbs
    让后说就写个简单的基于Sort的方法就行了,  忽略了非Sort的方法.




http://www.1point3acres.com/bbs/thread-205386-1-1.html
    2个面试官，1个career fair遇到的Asian姐姐, 另一个感觉是她的印度manager 鏉ユ簮涓€浜�.涓夊垎鍦拌鍧�.
问了很多C++的memory处理问题～
记得有stack overflow, how to write a garbage collector in C++ or java, where is reference created.鐣欏璁哄潧-涓€浜�-涓夊垎鍦�

coding:
spiral matrix~
记得多准备几种解法～


http://www.1point3acres.com/bbs/thread-207976-1-1.html

题目：输入n，求所有符合x^2+y^2+z^2=n的 x, y, z 组，要求O(n)的复杂度。
解法：遍历小于n的完全平方数，再用3sum。
求米。。. 鐣欏鐢宠璁哄潧-涓€浜╀笁鍒嗗湴




http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=295602&extra=page%3D43%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

很久之后，在地里面发面经，之前是申请，一晃我都要找工作了。。。
首先说下，本人upenn，Bloomberg的on-campus-recuriting，今天下午一点到两点点刚刚面完，一个小时
面我的是两个software engineer

上来自我介绍以及经典的why bb，老铁们可以好好准备一下
coding部分，纸上写代码，第一题是给一个tree，返回所有左叶子节点的和，这题divide and conquer的递归就可以做了，传入root和parent node，分析时间复杂度:O(n) 年代表树中的节点个数
第二题，给一个API叫做isSumK(Set s, int k) 判断当前的set中有没有subset的和是k，返回的是true和false；要求实现一个方法sumK(set s, int k) 通过调用isSumK来返回其中的一个解，就是返回一个subset的和是k; 我一上来说了dfs，但是被要求说，dfs的话，2的n次方时间复杂度特别高，且没有调用isSumK；在两人的提示下，大致出来的思路是，遍历元素，如果剩下的元素的和调用isSumK是true的话，就可以把当前元素抛弃掉，false的话，需要保留当前元素。。。以此减少问题的规模. 1point 3acres 璁哄潧

面之前看了地里面最新的面经，也刷过tag，但是被问到了两个比较新的题目（起码我没有在面经和tag中看到过），觉得有点遗憾吧，感觉是走远了，不过还是写着一篇面经，希望接下来要面试bb的人加油！希望对你们有帮助！也为之后的面试攒一把人品吧！加油啊，老铁们！



http://www.1point3acres.com/bbs/thread-295603-1-1.html
1v2, 稍微聊了一下简历，没问behavioral，两道利扣：一一七，四九八（起点改为右上角）498




http://www.1point3acres.com/bbs/thread-294919-1-1.html

第一次电面很水的两道Easy题，然后过了快两周，我以为挂了。HR又发信安排第二次电面。
我以为会考很难的题了这次，惴惴不安得快速翻阅以前做的Medium以上题的思路在昨天晚上。
结果一打开Hackerrank，又给我来一Easy题。利口特七反转整数。我故作镇定，说了一下思路。
然后一遍虫子自由得跑出结果123变321。写完后讨论了一些edge cases，比如负数啊溢出啊。.鏈枃鍘熷垱鑷�1point3acres璁哄潧
然后就是聊天，谈我做过的项目。
.1point3acres缃�
希望一路Easy到底，求昂赛和大米。




http://www.1point3acres.com/bbs/thread-271336-1-1.html


第一题, 一堆点汇集到主干, 可以看成从下往上的多叉树. 问所有点的最低公共祖先在哪里. 也就是所有点归集到那一个点的第一个点.-google 1point3acres

定义很模糊,就当输入是array of nodes吧,输出是node.

node class也没定义,不过肯定有parent这个属性,不确定可以商量..

我觉得,先求array size,也就是node个数.然后在每个node多加一个int属性,记录经过的node个数.第一个node的经过个数等于输入node个数时,就返回.. more info on 1point3acres.com

dfs的问题是一个点走太远.所以应该bfs,每个点走一遍,第一个node经过个数达标后直接返回.

这思路对不对? 可惜写bfs被误导到dfs..怪自己不够坚定..用英文讨论没有中文写下来这样清晰的思路更加不坚定,大家把我当前车之鉴.

-------. 鐗涗汉浜戦泦,涓€浜╀笁鍒嗗湴

第二题, 数据结构设计, 每个单位class记录的是名字,价格,id. 可以执行的操作是根据uid删除,添加一个新单位,根据价格返回top 5.


如果是堆加map,没见过在堆中间删一个node的.更何况是求top 5用堆也不方便. 那就是bst加map, 如果是这样,够坑的,当场写不出bst的插入,删除,取值,只能用bst的库.
. 鐗涗汉浜戦泦,涓€浜╀笁鍒嗗湴
更何况犯了贪心的错,根据经验总觉得添加或删除有一个能做到O(1).其实只有top 5是O(1)就够了吧.

. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷
这个方法求top5需要用到额外空间,应该有更好的解法..
--------. from: 1point3acres.com/bbs

其实布隆伯格的bar一直挺高的.没遇到lc原题,是运气问题,也是实力问题.



http://www.1point3acres.com/bbs/thread-292714-1-1.html
给一个起始数和目标数，只有两个运算（*2和/3），求从起始数到目标数的最短运算步数?解法：用BFS做，把每个数看成一个图上的一个点。



http://www.1point3acres.com/bbs/thread-243925-1-1.html

发个跪经，赞一波人品
找工作很晚，毕业了才开始刷题找工作，刚刚第一次电面，估计已经跪了，跪在了follow up， 问的很简单，lc的valid parenthese；
然后follow up 来了，需要加入单引号的判断，比如 " '[' ']' " 这样是valid， " '[ ]' " 这样不是valid，" ']' '[' "也不是valid，括号之间可以有其它的各种character，比如数字，字母， 一开始和面试官交流不太好，以为第二种也是valid，后来面试官说不是，所以就走了比较久的弯路，反应过来的时候时间快到了，不过面试官说思路是对的。
我的follow up思路是碰到 ' 直接和它后边的那个元素比较，如果也是 ' , 那么就是valid，如果不是，那么检测下一个元素是不是 ' ,然后中间的那个元素需要和stack顶端的元素匹配，成功则valid，不成功则不valid。
不知道说没说清楚。
.鏈枃鍘熷垱鑷�1point3acres璁哄潧
总之就是这些了，感觉和面试官交流很重要，代码也要非常熟练才行。



http://www.1point3acres.com/bbs/thread-177602-1-1.html

如题，挺紧张的，coding的时候感觉大脑不能思考，全靠女生缥缈的第六感在写，特别怕出bug。但是好在不难。转专业基础薄弱求放过。. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷
先说下time line， 2月底找人内推，3月1号收到电面邀请，3月10号电面。

面试官是个印度大哥， 有点口音，语速比较快，中间有些含糊的地方听得不太清楚。。。感觉特别严肃，一板一眼那种。。。听到口音时候我就有点不太好……
前面15分钟非常详细的问了一个简历上的project，边边角角的地方都很详细地问了是怎么实现的。

然后问了些数据结构的基础知识，开始coding，题目很简单，implement queue using stack. 先谈下思路，分析下复杂度，然后开始写。中间因为没用过codepair他给简单介绍了一下。
写的时候楼主很紧张大脑处于当机状态，不过好在跑test case的时候发现都过了。
然后大哥又继续开始问数据结构，详细问了下java的hashmap底层实现，各种极端情况的分析。当时想着大哥怎么没出第二题楼主内心挺着急，不是说起码两道coding才能过么。
讨论完之后，大哥又继续问了简历上面另外一个Project， 楼主就又介绍了一遍另外一个Project。讨论加介绍project总共就又花了大概15分钟。. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷

然后，然后总共45分钟的电面就进入了最后5分钟的提问环节。。。问了三四个常规问题，大哥答得也不是特别走心，笼统答了下然后就say goodbye了。。

电面就做了一道题，是不是要跪的节奏啊。求放过求放过啊
. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷
另外附上这两天总结的最近一两个月地里的电面面经，希望大家赏点大米，前段时间搜索花得有点狠。



//http://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=294922&extra=page%3D49%26filter%3Dsortid%26sortid%3D311%26sortid%3D311

早上的面试，没睡醒一大早就去了career center，坐着划了快十五分钟手机看到十来个人背着包拖着行李箱进来
又等了十五分钟等他们准备房间和分配，最后一个被叫去. 1point3acres.com/bbs
面我的是一个白人老头和国人姐姐，一开始白人老头看着有点严肃，国人姐非常nice地和我开场白
白人老头问了我简历上的project，那天感冒还没好喉咙不舒服，一开始加上紧张话都险些说不清楚。。。还不停咳嗽
project那边磕磕绊绊地说完了，老头又问一些很概念上的问题，系统设计会怎么做blabla
问题太宽泛只能也宽泛地回答了一下，后来我说完了老头还在等我继续说。。。略尴尬
然后老头就说好，给你出个题你回答一下，设计一个电梯

然后我在纸上边画uml diagram的草图边和他说，他问我这是uml吗，我说这只是一个草稿的uml，一些symbol比较随意，我设计的时候会这么做
然后讨论了怎么处理people和elevator的关系，然后他追问我这两个class之间的关系的时候没听懂他想问什么，来回了几次才说清楚
然后直接move on下一题了，国人姐姐出题
print binary tree，要求是从root开始每一个level从左到右print，每一层占独立的一行
用了两个queue，编程习惯问题。从loop内部开始写，结果在纸上不太好搞。。。稍显乱了一些，不知道是不是大早上的脑子有些不清醒。。。

最后问了一下bb工作环境以及两位的工作情况，最后老头和我说到收拾好去外面去等一下
我就和他们再次握了个手提了包出去，结果刚关上门老头就出来
和我说it was hopeful，HR过不久就会联系我，可能用电话的
一开始觉得自己表现不咋地，听到这句话我还内心感叹了一下这都行

总之就指着老头的最后那句话开始准备起onsite。。平时睡觉手机喜欢开飞行模式，为了等电话这几天都开大铃声保持开机睡觉
然后过了三天左右，手机突然跳出HR邮件，unfortunately, we have decided not to proceed blabla.
一开始有些郁闷，和说好的不一样啊，不过回头想来确实挂了也没有什么奇怪的. 涓€浜�-涓夊垎-鍦帮紝鐙鍙戝竷
只不过和去年巨水的电面相比体验上gap挺大的。
总之算是积累个经验吧，得寻找别的继续move on了。

*/
















//why bb



    public static void main(String []args){
        mergeData();
    }
}
