package medium;

import com.sun.org.apache.bcel.internal.generic.FADD;
import commons.Interval;
import commons.ListNode;
import commons.TreeLinkNode;

import java.util.*;

/**
 * Created by tao on 8/22/17.
 */


class UnionFind{
    private int n;
    private int[]parent=null;
    private int[]rank=null;
    public UnionFind(int nn){
        this.n=nn;
        rank = new int[nn+1];
        parent=new int[nn+1];
        for(int i=0;i<=n;++i){
            parent[i]=i;
        }
    }
    public int find(int x){
        while(parent[x]!=x){
            parent[x]=parent[parent[x]];
            x=parent[x];
        }
        return x;
    }


    public boolean union(int x,int y){
        int xx = find(x);
        int yy = find(y);
        if(xx==yy)
            return false;
        if(rank[xx]<rank[yy]){
            parent[xx]=yy;
        }else if(rank[xx]>rank[yy]){
            parent[yy]=xx;
        }else{
            parent[xx]=yy;
            rank[yy]++;
        }

        return true;
    }

    public boolean connected(int x,int y){
        return find(x)==find(y);
    }
}

public class MediumQuestions {

    public int lengthOfLongestSubstring(String s){
        int n= s.length(),begin=0,end=0;
        int d=0,maxLen=0;
        int []cnt=new int[128];
        while(end<n){
            if(cnt[s.charAt(end++)]++ ==1)
                d++;
            while(d>=1){
                if(--cnt[s.charAt(begin++)]==1)
                    d--;
            }
            maxLen=Math.max(maxLen,end-begin);
        }
        return maxLen;
    }

    //4 median of two sorted arrays
    public int findKth(int[]nums1,int start1,int end1,int []nums2,int start2,int end2,int k){
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
        int first = findKth(nums1,0,nums1.length,nums2,0,nums2.length,m/2 +1);
        if(m%2!=0)
            return 1.0*first;
        else return 1.0*(findKth(nums1,0,nums1.length,nums2,0,nums2.length,m/2)+first)/2.0;
    }

    public double findMedianSorted(int[]nums1,int[]nums2){
        int m = nums1.length,n=nums2.length;
        if(m>n)
            return findMedianSorted(nums2,nums1);
        int min=0,max=m,half =(m+n+1)/2;
        int maxLeft=0,minRight=0;
        while(min<=max){
            int i =(max-min)/2+min;
            int j = half-i;
            if(i<m && nums2[j-1]>nums1[i])
                min=i+1;
            else if(i>0 && nums1[i-1]>nums2[j])
                max=i-1;
            else{
                if(i==0)
                    maxLeft=nums2[j-1];
                else if(j==0)
                    maxLeft=nums1[i-1];
                else
                    maxLeft=Math.max(nums1[i-1],nums2[j-1]);
                if((m+n)%2==1)
                    return 1.0*maxLeft;
                if(i==m)
                    minRight=nums2[j];
                else if(j==n)
                    minRight=nums1[i];
                else
                    minRight=Math.min(nums1[i],nums2[j]);
                return (maxLeft+minRight)/2.0;
            }

        }
        return 0.0;
    }


    //15 3 sum
    public List<List<Integer>>threeSum(int[]nums){
        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>>res=new ArrayList<>();
        for(int i=0;i<n-2;++i){
            if(i>0 && nums[i]==nums[i-1])
                continue;
            int begin=i+1,end=n-1;
            while(begin<end){
                int sum = nums[begin]+nums[i]+nums[end];
                if(sum==0){
                    res.add(Arrays.asList(nums[i],nums[begin],nums[end]));
                    begin++;
                    end--;
                    while(begin<end && nums[begin]==nums[begin-1])
                        begin++;
                    while(begin<end && nums[end]==nums[end+1])
                        end--;
                }else if(sum<0)
                    begin++;
                else
                    end--;
            }
        }
        return res;
    }



    public int threeSumCloest(int[]nums,int target){
        int n = nums.length;
        Arrays.sort(nums);
        int val = nums[0]+nums[1]+nums[2];
        for(int i=0;i<n-2;++i){
            if(i>0 && nums[i]==nums[i-1])
                continue;
            int begin =i+1;
            int end=n-1;
            while(begin<end){
                int sum = nums[i]+nums[begin]+nums[end];
                if(Math.abs(sum-target)<Math.abs(val-target))
                    val=sum;
                if(sum>target)
                    end--;
                else if(sum<target)
                    begin++;
                else
                    return target;
            }

        }
        return val;
    }



    //17 letter combinations of a phone number
    public void dfs(String digits,List<String>res,String[]args,String path,int index){
        if(index==digits.length() && path.length()==digits.length()){
            res.add(path);
            return;
        }
        char []cc = args[digits.charAt(index)-'2'].toCharArray();
        for(char c:cc){
            dfs(digits,res,args,path+c,index+1);
        }
    }
    public List<String>letterCombinations(String digits){
        List<String>res=new ArrayList<>();
        String[]args={"abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
        dfs(digits,res,args,"",0);
        return res;
    }


    public ListNode mergeKLists(ListNode[]lists){
        ListNode dummy = new ListNode(0);
        ListNode p = dummy;
        PriorityQueue<ListNode>pq=new PriorityQueue<>(new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return o1.val-o2.val;
            }
        });
        for(ListNode list:lists){
            if(list!=null)
                pq.offer(list);
        }
        while(!pq.isEmpty()){
            ListNode top = pq.poll();
            p.next=top;
            p=p.next;
            if(top.next!=null)
                pq.offer(top.next);
        }
        return dummy.next;
    }


    public ListNode swapPairs(ListNode head){
        //iterative way
        if(head==null||head.next==null)
            return head;
        ListNode dummy=head.next;
        ListNode first = head;
        ListNode second = head.next;
        ListNode pre =null;
        while(first!=null && second!=null){
            ListNode node = second.next;
            second.next=first;
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


    public int longestValidParentheses(String s) {
        char []ss=s.toCharArray();
        int n = ss.length,maxSize=0;
        int []dp=new int[n+1];
        for(int i=2;i<=n;++i){
            if(ss[i-1]==')'){
                if(ss[i-2]=='('){
                    dp[i]=2+dp[i-2];
                    maxSize=Math.max(maxSize,dp[i]);
                }else{
                    int ind = i-dp[i-1]-1;
                    if(ind>=1 && ss[ind-1]=='('){
                        dp[i]=dp[i-1]+2+dp[ind-1];
                        maxSize=Math.max(maxSize,dp[i]);
                    }
                }
            }
        }
        return maxSize;
    }


    public int[] searchRange(int[] nums, int target) {
        //one binary search
        int n = nums.length;
        int[]res={-1,-1};
        int begin=0,end=n-1;
        while(begin<=end){
            int mid =(end-begin)/2+begin;
            if(nums[mid]==target){
                //two directions
                //search lowrbound
                int l=0,r=mid;
                while(l<r){
                    int mid1=(r-l)/2+l;
                    if(nums[mid1]==target)
                        r=mid1;
                    else
                        l=mid1+1;
                }
                res[0]=l;


                //search upperbound
                l=mid;r=n-1;
                while(l<r){
                    int mid1=(r-l)/2+l;
                    if(nums[mid1]==target)
                        l=mid1+1;
                    else
                        r=mid;
                }
                if(nums[l]>target)
                    l--;
                res[1]=l;
                break;
            }else if(nums[mid]>target)
                end=mid-1;
            else
                begin=mid+1;
        }
        return res;
    }


    //binary search
    public int binarySearch(int[]nums,int target){
        int n = nums.length;
        int begin=0,end=n-1;
        while(begin<=end){
            int mid=(end-begin)/2+begin;
            if(nums[mid]==target)
                return mid;
            else if(nums[mid]>target)
                end=mid-1;
            else
                begin=mid+1;
        }
        return -1;
    }

    public String multiply(String num1, String num2) {
        //brute force
        int m=num1.length(),n=num2.length();
        char[]nums1=num1.toCharArray();
        char[]nums2=num2.toCharArray();
        char[]ss=new char[m+n];
        Arrays.fill(ss,'0');
        for(int i=m-1;i>=0;--i){
            int carry=0;
            for(int j=n-1;j>=0;--j){
                int sum=(ss[i+j+1]-'0')+carry+(nums1[i]-'0')*(nums2[j]-'0');
                ss[i+j+1]=(char)('0'+sum%10);
                carry=sum/10;
            }
            if(carry!=0)
                ss[i]=(char)(ss[i]+carry);
        }
        //remove the lead 0
        int ind=0;
        while(ind<m+n-1){
            if(ss[ind]!='0')
                break;
            ind++;
        }
        return String.valueOf(ss).substring(ind);
    }

    public int jump(int[] nums) {
        //greedy
        //jump as far as possibe
        int reach=0,n=nums.length;
        if(n<=1)
            return 0;
        int minStep=0;
        int index=0;
        while(reach<n-1){
            int optimalIndex=index;
            int sum=0;
            if(index>=n)
                break;
            for(int i=1;i<=nums[index];++i){
                if(index+i>=n-1){
                    optimalIndex=n;
                    break;
                }
                if(index+i+nums[index+i]>sum){
                    sum=index+i+nums[index+i];
                    optimalIndex=index+i;
                }
            }
            minStep++;
            reach=optimalIndex;
            index=optimalIndex;
        }
        return minStep;
    }


    public List<Interval> merge(List<Interval> intervals) {
        List<Interval>res=new ArrayList<>();
        int n=intervals.size();
        if(n==0)
            return res;
        intervals.sort(new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return o1.start-o2.start;
            }
        });
        res.add(intervals.get(0));
        for(int i=1;i<n;++i){
            Interval ee = res.get(res.size()-1);
            if(intervals.get(i).start>ee.end)
                res.add(intervals.get(i));
            else{
                ee.end=Math.max(ee.end,intervals.get(i).end);
                res.set(res.size()-1,ee);
            }
        }
        return res;
    }


    public boolean isNumber(String s) {
        char[]ss=s.toCharArray();
        int end = ss.length;
        int n=ss.length;
        int begin=0;
        //delete space
        while(begin<n && Character.isSpaceChar(ss[begin])){
            begin++;
        }

        //
        if(begin<end && (ss[begin]=='-'||ss[begin]=='+')){
            begin++;
        }

        //number
        int num=0;
        while(begin<end && Character.isDigit(ss[begin])){
            begin++;
            num++;
        }
        if(begin<end && ss[begin]=='.'){
            begin++;
            while(begin<end && Character.isDigit(ss[begin]))
                begin++;
        }
        boolean hasE=false;
        int index=0;
        if(begin<end && (ss[begin]=='E'||ss[begin]=='e')){
            if(num<1)
                return false;
            begin++;
            hasE=true;
            if(begin<end && (ss[begin]=='-'||ss[begin]=='+')){
                begin++;
            }
            while(begin<end && Character.isDigit(ss[begin])){
                begin++;
                index++;
            }
        }

        while(begin<end && Character.isSpaceChar(ss[begin]))
            begin++;
        if(num<1||begin!=end)
            return false;
        if(hasE && index<1)
            return false;
        return true;


    }


    public void setZeroes(int[][] matrix) {
        //第一行存储各列是否有0
        //第一列存储各行是否有0,,共用了一个matrix[0][0],所以需要一个额外的变量
        if(matrix.length==0||matrix[0].length==0)
            return;
        int m=matrix.length,n=matrix[0].length;
        boolean firstRow=false;
        for(int i=0;i<m;++i){
            if(matrix[i][0]==0){
                firstRow=true;
                break;
            }
        }
        for(int i=0;i<m;++i){
            for(int j=1;j<n;++j){
                if(matrix[i][j]==0){
                    matrix[i][0]=0;
                    matrix[0][j]=0;
                }
            }
        }

        //bottom to up
        for(int i=m-1;i>=0;--i){
            for(int j=n-1;j>=1;--j){
                if(matrix[0][j]==0||matrix[i][0]==0)
                    matrix[i][j]=0;
            }
            if(firstRow)
                matrix[i][0]=0;
        }
    }


    //68 text justification
    public List<String>fullJustify(String[]words,int maxWidth){
        int n=words.length;
        List<String>res = new ArrayList<>();
        if(n==0||maxWidth==0)
            return res;
        int w =0;
        for(int i=0;i<n;i=w){
            int len=-1;
            for(w=i;w<n && len+1+words[w].length()<=maxWidth;++w){
                len+=1+words[w].length();
            }
            int gaps=w-i-1;
            int evenSpace =1;
            int extraSpace =0;
            StringBuilder str = new StringBuilder(words[i]);
            if(w!=i+1 && w!=n){//not single string or last line
                evenSpace=(maxWidth-len)/gaps+1;
                extraSpace=(maxWidth-len)%gaps;
            }
            for(int j=i+1;j<w;++j){
                for(int s=0;s<evenSpace;++s)
                    str.append(' ');
                if(extraSpace>0){
                    str.append(' ');
                    extraSpace--;
                }
                str.append(words[j]);
            }

            int remaing=maxWidth-str.length();
            while(remaing>0){
                str.append(' ');
                remaing--;
            }
            res.add(str.toString());
        }
        return res;
    }



    public void connectII(TreeLinkNode root){
        TreeLinkNode dummy = new TreeLinkNode(0);
        dummy.next=root;
        TreeLinkNode node = root;
        while(dummy.next!=null){
            node=dummy.next;
            dummy.next=null;
            TreeLinkNode head=dummy;
            for(;node!=null;node=node.next){
                if(node.left!=null){
                    head.next=node.left;
                    head=head.next;
                }
                if(node.right!=null){
                    head.next=node.right;
                    head=head.next;
                }
            }
        }
        System.out.println("finish");
    }

    public boolean isPalindrome(String s) {
        char[]ss=s.toCharArray();
        int n = ss.length;
        int left=0,right=n-1;
        while(left<right){
            while(left<right && !Character.isLetterOrDigit(left))
                left++;
            while(left<right && !Character.isLetterOrDigit(right))
                right--;
            if(left<right){
                if(Character.toUpperCase(ss[left])!=Character.toUpperCase(ss[right]))
                    return false;
                left++;
                right--;
            }
        }
        return true;
    }


    public void solve(char[][] board) {
        //quque
        if(board.length==0||board[0].length==0)
            return;
        int m = board.length,n=board[0].length;
        int[]dx={1,-1,0,0};
        int []dy={0,0,1,-1};
        boolean [][]vis=new boolean[m][n];
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(!vis[i][j]&&board[i][j]=='O'){
                    Queue<int[]>q=new LinkedList<>();
                    Queue<int[]>q1=new LinkedList<>();
                    boolean filled=true;
                    q.offer(new int[]{i,j});
                    q1.offer(new int[]{i,j});
                    vis[i][j]=true;
                    while(!q.isEmpty()){
                        int []top=q.poll();
                        for(int k=0;k<4;++k){
                            int nx = top[0]+dx[k];
                            int ny = top[1]+dy[k];
                            if(nx<0||ny<0||nx>=m||ny>=n||vis[nx][ny]||board[nx][ny]!='O')
                                continue;
                            if(nx==0||ny==0||nx==m-1||ny==n-1)
                                filled=false;
                            vis[nx][ny]=true;
                            q.offer(new int[]{nx,ny});
                            q1.offer(new int[]{nx,ny});

                        }
                    }
                    if(filled){
                        while(!q1.isEmpty()){
                            int []top=q1.poll();
                            board[top[0]][top[1]]='X';
                        }
                    }
                }
            }
        }
    }


    public void solve1(char[][] board) {
        //边界就连接到m*n
        if(board.length==0||board[0].length==0)
            return ;
        int m = board.length,n=board[0].length;
        StringBuilder sb = new StringBuilder();
        UnionFind uf = new UnionFind(m*n);
        //boolean [][]vis=new boolean[m][n];
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(board[i][j]=='O'){
                    int index = i*n+j;
                    //vis[i][j]=true;
                    if(i==0||i==m-1||j==0||j==n-1)
                        uf.union(index,m*n);

                    //four directions
                    if(i>0 && board[i-1][j]=='O'){
                        //vis[i-1][j]=true;
                        uf.union(index,index-n);

                    }
                    if(j>0 && board[i][j-1]=='O'){
                        //vis[i][j-1]=true;
                        uf.union(index,index-1);

                    }
                    if(i<m-1 && board[i+1][j]=='O' ){
                        //vis[i+1][j]=true;
                        uf.union(index,index+n);

                    }
                    if(j<n-1 && board[i][j+1]=='O'){
                        //vis[i][j+1]=true;
                        uf.union(index,index+1);

                    }
                }
            }
        }
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(board[i][j]=='O'&& !uf.connected(m*n,i*n+j)){
                    board[i][j]='X';
                }
            }
        }
    }


    public int minCut(String s) {
        int n = s.length();
        int []dp=new int[n];
        boolean [][]isPalindrome=new boolean[n][n];
        for(int i=0;i<n;++i){
            dp[i]=Math.max(i-1,0);
            isPalindrome[i][i]=true;
            for(int j=i;j>=0;--j){
                if(s.charAt(i)==s.charAt(j) && (i-1<=j+1||isPalindrome[j+1][i-1])){
                    isPalindrome[j][i]=true;
                    if(j==0)
                        dp[i]=0;
                    else
                        dp[i]=Math.min(dp[i],dp[j-1]+1);
                }
            }
        }

        return dp[n-1];
    }

    public ListNode insertionSortList(ListNode head) {
        if(head==null||head.next==null)
            return head;
        ListNode dummy = new ListNode(Integer.MIN_VALUE);
        ListNode node = head;
        while(node!=null){
            ListNode p = dummy;
            ListNode nextNode=node.next;
            node.next=null;
            while(p!=null && p.next!=null && node.val>=p.val){
                if(p.next!=null && p.next.val>node.val)
                    break;
                p=p.next;
            }
            if(p.next!=null && p.next.val>node.val){
                node.next=p.next;
                p.next=node;
            }else{
                p.next=node;
            }
            node=nextNode;
        }
        return dummy.next;
    }


    public int maxProduct(int[] nums) {
        //
        int maxVal=1,minVal=1,res=Integer.MIN_VALUE;
        for(int x:nums){
            int tmpMax=maxVal;
            maxVal=Math.max(Math.max(x,x*maxVal),x*minVal);
            minVal=Math.min(Math.min(x,x*tmpMax),x*minVal);
            res=Math.max(maxVal,res);
        }
        return res;
    }

    public String largestNumber(int[] nums) {
        //sort
        List<Integer>res=new ArrayList<>();
        for(int x:nums)
            res.add(x);
        res.sort(new Comparator<Integer>() {
            @Override
            public int compare(Integer num1, Integer num2) {
                int n1=Integer.parseInt(""+num1+num2);
                int n2=Integer.parseInt(""+num2+num1);
                if(n1<n2)
                    return 1;
                else if(n1>n2)
                    return -1;
                else
                    return 0;
            }
        });
        StringBuilder sb=new StringBuilder();
        for(int x:nums){
            sb.append(x);
        }
        return sb.toString();
    }

    public void reverse(char[]s,int begin,int end){
        while(begin<end){
            char c = s[begin];
            s[begin++]=s[end];
            s[end--]=c;
        }

    }
    public void reverseWords(char[] s) {
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
        System.out.println(String.valueOf(s));

    }

    public int maximalSquare(char[][] matrix) {
        //save space
        if(matrix.length==0||matrix[0].length==0)
            return 0;
        int m = matrix.length,n=matrix[0].length,maxLen=0;
        int []dp=new int[n];
        for(int i=0;i<n;++i){
            if(matrix[0][i]=='1'){
                dp[i]=1;
                maxLen=1;
            }
        }

        for(int i=1;i<m;++i){
            int[]pre=dp.clone();
            Arrays.fill(dp,0);
            dp[0]=(matrix[i][0]=='1'?1:0);
            maxLen=Math.max(maxLen,dp[0]);
            for(int j=1;j<n;++j){
                if(matrix[i][j]=='1'){
                    dp[j]=Math.min(dp[j-1],Math.min(pre[j],pre[j-1]))+1;
                    maxLen=Math.max(maxLen,dp[j]);
                }
            }
        }
        return maxLen*maxLen;
    }

    public int calculate(String s) {
        Stack<Integer>stk=new Stack<>();//store the sign and value;
        int val=0,sign=1,res=0;
        int i=0,n=s.length();
        char []ss=s.toCharArray();
        while(i<n){
            if(ss[i]==' '){
                i++;
            }
            else if(Character.isDigit(ss[i])){
                val=10*val+ss[i]-'0';
                i++;
            }else if(ss[i]=='+'){
                res+=sign*val;
                val=0;
                sign=1;
                i++;
            }else if(ss[i]=='-'){
                res+=sign*val;
                sign=-1;
                val=0;
                i++;
            }else if(ss[i]=='('){
                //push to stack
                stk.push(res);
                stk.push(sign);
                res=0;
                sign=1;
                i++;
            }else if(ss[i]==')'){
                //pop out to stack
                res+=sign*val;
                val=0;
                int negative=stk.pop();
                res=negative*res+stk.pop();
                i++;
            }
        }
        return res+sign*val;
    }


    public void deal(Stack<Integer>stk,boolean hasSlash,boolean hasStar,int sign,int num){
        if(hasStar){
            int top = stk.pop();
            stk.push(sign*num*top);
        }
        else if(hasSlash){
            int top = stk.pop();
            stk.push(top/(sign*num));
        }else
            stk.push(sign*num);
    }
    public int calculateII(String s){
        s+='+';
        char []ss=s.toCharArray();
        Stack<Integer>stk=new Stack<>();
        int n = ss.length,num=0;
        int sign=1;
        boolean hasStar=false;
        boolean hasSlash=false;
        for(int i=0;i<n;++i){
            if(ss[i]==' '){

            }else if(ss[i]=='+'){
                deal(stk,hasSlash,hasStar,sign,num);
                hasSlash=hasStar=false;
                sign=1;
                num=0;
            }else if(ss[i]=='-'){
                deal(stk,hasSlash,hasStar,sign,num);
                hasSlash=hasStar=false;
                sign=-1;
                num=0;
            }else if(Character.isDigit(ss[i])){
                num=10*num+(ss[i]-'0');
            }else if(ss[i]=='*'||ss[i]=='/'){
                deal(stk,hasSlash,hasStar,sign,num);
                hasSlash=hasStar=false;
                num=0;
                sign=1;
                if(ss[i]=='*')
                    hasStar=true;
                else
                    hasSlash=true;
            }
        }
        int sum=0;
        while(!stk.isEmpty()){
            sum+=stk.pop();
        }
        return sum;
    }



    public int countDigitOne(int n) {
        int[]dp=new int[10];
        int []sum=new int[10];
        dp[1]=1;
        sum[1]=1;
        for(int i=2;i<10;++i){
            dp[i]=(int)Math.pow(10,i-1)+9*(i-1);
            sum[i]=dp[i]+sum[i-1];
        }
        return countDigitOne(n,dp,sum);
    }
    public int countDigitOne(int x,int[]dp,int[]sum){
        if(x==0||x==1)
            return x;

        return 0;
    }


    public List<Integer> diffWaysToCompute(String input) {
        List<Integer>res=new ArrayList<>();
        if(input==null||input.isEmpty()){
            res.add(0);
            return res;
        }
        boolean hasOpera=false;
        for(int i=0;i<input.length();++i){
            char c = input.charAt(i);
            if(c=='+'||c=='-'||c=='*'){
                hasOpera=true;
                List<Integer>left = diffWaysToCompute(input.substring(0,i));
                List<Integer>right=diffWaysToCompute(input.substring(i+1));
                for(int l = 0;l<left.size();++l){
                    for(int r =0;r<right.size();++r){
                        if(c=='+'){
                            res.add(left.get(l)+right.get(r));
                        }else if(c=='-'){
                            res.add(left.get(l)-right.get(r));
                        }else{
                            res.add(left.get(l)*right.get(r));
                        }
                    }
                }
                //return res;
            }
        }
        if(!hasOpera)
            res.add(Integer.parseInt(input));
        return res;
    }


    public void backtracking(List<List<Integer>>res,int[]nums,List<Integer>path){
        if(path.size()==res.size()){
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i=0;i<nums.length;i++){
            if(!path.contains(nums[i])){
                path.add(nums[i]);
                backtracking(res,nums,path);
                path.remove(path.size()-1);
            }
        }
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>>res=new ArrayList<>();
        List<Integer>path=new ArrayList<>();
        backtracking(res,nums,path);
        return res;
    }


    public boolean backtracking(long first,long second,String num){
        if(num.isEmpty())
            return true;
        long third = first+second;
        String thirdStr = String.valueOf(third);
        if(num.startsWith(thirdStr)){
            if(backtracking(second,third,num.substring(thirdStr.length())))
                return true;
        }
        return false;
    }
    public boolean isAdditiveNumber(String num) {
        //try backtracking
        int n = num.length();
        for(int i=1;i<=n/2;++i){
            String first = num.substring(0,i);
            long firstNum = Long.parseLong(first);
            if(String.valueOf(firstNum).length()!=first.length())
                return false;
            for(int j=i+1;j<=i+(n-i)/2;++j){
                String second = num.substring(i,j);
                long secondNum = Long.parseLong(second);
                if(String.valueOf(secondNum).length()!=second.length())
                    break;
                if(backtracking(firstNum,secondNum,num.substring(j)))
                    return true;
            }
        }
        return false;
    }



    public String removeDuplicateLetters(String s) {
        char []ss=s.toCharArray();
        int []cnt=new int[26];
        int n = ss.length;
        for(char c:ss)
            cnt[c-'a']++;
        boolean []vis=new boolean[26];
        Stack<Integer>stk=new Stack<>();
        for(int i=0;i<n;++i){
            cnt[ss[i]-'a']--;
            if(vis[ss[i]-'a'])
                continue;
            while(!stk.isEmpty() && cnt[ss[stk.peek()]-'a']>0 && ss[stk.peek()]>ss[i]){
                int ind = stk.pop();
                vis[ss[ind]-'a']=false;
            }
            stk.push(i);
            vis[ss[i]-'a']=true;
        }
        StringBuilder sb = new StringBuilder();
        while(!stk.isEmpty()){
            sb.append(ss[stk.pop()]);
        }
        sb.reverse();
        return sb.toString();
    }

    public ListNode oddEvenList(ListNode head) {
        if(head==null||head.next==null)
            return head;
        ListNode odd=head;
        ListNode even=head.next;
        ListNode podd=odd;
        ListNode peven=even;
        head=head.next.next;
        while(head!=null){
            podd.next=head;
            peven.next=head.next;
            podd=podd.next;
            peven=peven.next;
            if(head.next!=null)
                head=head.next.next;
        }
        podd.next=even;
        peven.next=null;
        return podd;
    }


    public int dfs(int[][]matrix,int i,int j,boolean[][]vis){
        if(i>=matrix.length||j>=matrix[0].length||i<0||j<0||vis[i][j])
            return 0;
        vis[i][j]=true;
        int ans=1;
        if(i>=1 && matrix[i-1][j]>matrix[i][j])
            ans=Math.max(ans,1+dfs(matrix,i-1,j,vis));

        if(j>=1 && matrix[i][j-1]>matrix[i][j])
            ans=Math.max(ans,1+dfs(matrix,i,j-1,vis));

        if(i<matrix.length-1 && matrix[i+1][j]>matrix[i][j])
            ans=Math.max(ans,1+dfs(matrix,i+1,j,vis));

        if(j<matrix[0].length-1 && matrix[i][j+1]>matrix[i][j])
            ans=Math.max(ans,1+dfs(matrix,i,j+1,vis));
        vis[i][j]=false;
        return ans;
    }
    public int longestIncreasingPath(int[][] matrix) {
        int res=0;
        if(matrix.length==0||matrix[0].length==0)
            return 0;
        int m = matrix.length,n=matrix[0].length;
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                boolean [][]vis=new boolean[m][n];
                int x = dfs(matrix,i,j,vis);
                if(x>res)
                    res=x;
            }
        }
        return res;
    }

    public int minPatches(int[] nums, int n) {
        int nn =nums.length;
        long missing=1;
        int cnt=0,i=0;
        while(missing<=n){
            if(i<nn && missing>=nums[i])
                missing+=nums[i++];
            else{
                missing<<=1;
                cnt++;
            }
        }
        return cnt;
    }

    public int greater(int[]nums1,int[]nums2){
        int n =nums1.length;
        for(int i=0;i<n;++i){
            if(nums1[i]>nums2[i])
                return 1;
            else if(nums1[i]<nums2[i])
                return -1;
        }
        return 0;
    }
    public int[]getMax(int[]nums,int n){
        //get n numbers, and make its maximum
        Stack<Integer>stk=new Stack<>();
        int[]res=new int[n];
        int ind=n-1,m=nums.length;
        for(int i=0;i<m;++i){
            while(!stk.isEmpty() && nums[i]>stk.peek() && (n-stk.size())<(m-i)){
                stk.pop();
            }
            if(stk.size()<n)
                stk.push(nums[i]);
        }
        while(!stk.isEmpty()){
            res[ind--]=stk.pop();
        }
        for(int x:res)
            System.out.println(x);
        return res;
    }

    public int[]combine(int[]nums1,int[]nums2){
        int m=nums1.length,n=nums2.length;
        int ind=0;
        int[]res=new int[m+n];
        while(ind<m){
            res[ind]=nums1[ind];
            ind++;
        }
        while(ind<m+n){
            res[ind]=nums2[ind-m];
            ind++;
        }
        return res;
    }
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int m = nums1.length,n=nums2.length;
        //分配i个给nums1, k-i个给nums2
        int[]res=new int[k];
        for(int i=Math.max(0,k-n);i<=Math.min(k,m);++i){
            int []res1=getMax(nums1,i);
            int []res2=getMax(nums2,k-i);
            int []tmp1=combine(res1,res2);
            int []tmp2=combine(res2,res1);
            if(greater(tmp1,res)>0)
                res=tmp1.clone();
            if(greater(tmp2,res)>0)
                res=tmp2.clone();
        }
        return res;
    }


    public boolean deletePound(Stack<String>stk){
        while(stk.size()>=2){
            String first = stk.pop();
            if(!first.equals("#")){
                stk.push(first);
                break;
            }
            if(stk.peek().equals("#")){
                stk.pop();
            }else{
                stk.push(first);
                break;
            }
            if(!stk.isEmpty()){
                if(stk.pop().equals("#"))
                    return false;
                stk.push("#");
            }else
                return false;
        }
        return true;
    }
    public boolean isValidSerialization(String preorder) {
        Stack<String>stk=new Stack<>();
        String[]args=preorder.split(",");
        for(String str:args){
            if(!stk.isEmpty() && stk.peek().equals("#") && str.equals("#")){
                stk.push("#");
                if(!deletePound(stk))
                    return false;
            }else{
                stk.push(str);
            }
        }
        if(!deletePound(stk))
            return false;
        return stk.size()==1 && stk.peek().equals("#");

    }
    public boolean dfs(List<String>res,Map<String,List<Terminal>>neighbors,String start,int len){
        res.add(start);
        if(res.size()==len){
            return true;
        }
        List<Terminal>terminals=neighbors.get(start);
        int size=terminals.size();
        for(int i=0;i<size;++i){
            if(!terminals.get(i).visited){
                terminals.get(i).visited=true;
                if(dfs(res,neighbors,terminals.get(i).name,len))
                    return true;
                terminals.get(i).visited=false;
            }
        }
        res.remove(res.size()-1);
        return false;
    }
    class Terminal{
        String name;
        boolean visited;
        public Terminal(String name,boolean vised){
            this.name=name;
            this.visited=vised;
        }
    }
    public List<String> findItinerary(String[][] tickets) {
        Map<String,List<Terminal>>neighbors=new HashMap<>();
        for(String[]ticket:tickets){
            if(!neighbors.containsKey(ticket[0]))
                neighbors.put(ticket[0],new ArrayList<>());
            neighbors.get(ticket[0]).add(new Terminal(ticket[1],false));
        }
        for(Map.Entry<String,List<Terminal>>entry:neighbors.entrySet()){
            List<Terminal>terminal=entry.getValue();
            Collections.sort(terminal, new Comparator<Terminal>() {
                @Override
                public int compare(Terminal o1, Terminal o2) {
                    return o1.name.compareTo(o2.name);
                }
            });
        }
        List<String>res=new ArrayList<>();
        dfs(res,neighbors,"JFK",tickets.length+1);
        return res;
    }

    public int getMax(int []nums){
        int maxValue=Integer.MIN_VALUE;
        int curSum=0;
        for(int x:nums){
            curSum+=x;
            if(maxValue<curSum)
                maxValue=curSum;
            if(curSum<0)
                curSum=0;
        }
        return maxValue;
    }
    public int getMax(int[][]matrix){
        int m = matrix.length,n=matrix[0].length;
        int maxVal=Integer.MIN_VALUE;
        for(int l=0;l<n;++l){
            int[]colSum=new int[m];
            for(int r=l;r<n;++r){
                for(int i=0;i<m;++i)
                    colSum[i]+=matrix[i][r];
                int val = getMax(colSum);
                if(maxVal<val)
                    maxVal=val;
            }
        }
        return maxVal;
    }


    public boolean isPerfectSquare(int num) {
        //sqrt
        double first=1.0*num;
        while(Math.abs(first*first-num)>0.0001){
            first = (first+num/first)/2.0;
            System.out.println(first);
        }
        int res = (int)first;
        System.out.println(res);
        return res*res==num;
    }




    public int maxKilledEnemies(char[][] grid) {
        if(grid.length==0||grid[0].length==0)
            return 0;
        int m = grid.length,n=grid[0].length;
        int row=0,maxNum=0;
        int []col=new int[n];
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(grid[i][j]=='W')
                    continue;
                if(j==0||grid[i][j-1]=='W'){
                    int x =j;
                    row=0;
                    while(x<n && grid[i][x]!='W'){
                        if(grid[i][x]=='E')
                            row++;
                        x++;
                    }
                }
                if(i==0||grid[i-1][j]=='W'){
                    int x = i;
                    col[j]=0;
                    while(x<m && grid[x][j]!='W'){
                        if(grid[x][j]=='E')
                            col[j]++;
                        x++;
                    }
                }
                if(grid[i][j]=='0')
                    maxNum=Math.max(maxNum,row+col[j]);
            }

        }
        return maxNum;
    }

    public int quickPow(int a,int b){
        int res=1;
        while(b>0){
            if((b&0x1)!=0)
                res=((res%1337)*(a%1337))%1337;
            a=((a%1337)*(a%1337)%1337);
            b>>=1;
        }
        return res;
    }
    public int func(int a,int []b,int end){
        if(end==0)
            return quickPow(a,b[0]);
        int first = quickPow(a,b[end]);
        int second = func(a,b,end-1);
        second=quickPow(second,10);
        return ((first%1337)*(second%1337))%1337;
    }
    public int superPow(int a, int[] b) {
        return func(a,b,b.length-1);
    }

    public boolean hasCycle(Map<Integer,List<Integer>>map,int start,boolean[]vis,boolean[]onLoop){
        if(vis[start])
            return false;
        if(onLoop[start])
            return true;
        onLoop[start]=true;
        List<Integer>edges=map.getOrDefault(start,new ArrayList<>());
        for(int edge:edges){
            if(hasCycle(map,edge,vis,onLoop))
                return true;
        }
        vis[start]=true;
        return false;
    }
    public int[] findRedundantDirectedConnection(int[][] edges) {
        int n = edges.length;
        int ind = n-1;
        for(int i=n-1;i>=0;--i){
            //build map again
            Map<Integer,List<Integer>>map = new HashMap<>();
            for(int j=n-1;j>=0;--j){
                if(j==ind)
                    continue;
                if(!map.containsKey(edges[j][0]))
                    map.put(edges[j][0],new ArrayList<>());
                map.get(edges[j][0]).add(edges[j][1]);
            }
            boolean[]vis=new boolean[n+1];
            boolean has = false;
            for(int ii=1;ii<=n;++ii){
                if(hasCycle(map,ii,vis,new boolean[n+1])){
                    has = true;
                    break;
                }
            }
            if(!has)
                return edges[ind];
            ind--;
        }
        int []res = {0,0};
        return res;
    }

    public boolean valid(char[]ss){
        return Integer.parseInt(String.valueOf(ss[0]+""+ss[1]))<=23 && Integer.parseInt(String.valueOf(ss[2]+""+ss[3]))<=59;
    }
    public String nextClosestTime(String time) {
        char []digits = (time.substring(0,2)+time.substring(3,5)).toCharArray();
        char []times = digits.clone();
        Arrays.sort(digits);
        for(int i=3;i>=0;--i){
            char c = times[i];
            for(int j =0;j<=3;++j){
                if(digits[j]>times[i]){
                    times[i]= digits[j];
                    if(valid(times))
                        return String.valueOf(times[0]+""+times[1])+":"+String.valueOf(times[2]+""+times[3]);
                    times[i]= c;
                    break;
                }
            }
        }
        return String.valueOf(digits[0]+""+digits[0])+":"+String.valueOf(digits[0]+""+digits[0]);
    }


    public int maxA(int N) {
        int []dp= new int[N+1];
        int []v =new int[N+1];
        int []copy = new int[N+1];
        for(int i=1;i<=N;++i){
            dp[i]=Math.max(dp[i],dp[i-1]+1);
            if(i>=3){
                dp[i] = Math.max(dp[i],2*dp[i-3]);
                v[i]=dp[i-3];
            }
            for(int j=i-4;j>=0;--j)
                dp[i] = Math.max(dp[i],(i-1-j)*dp[j]);
        }
        return dp[N];
    }

    public String predictPartyVictory(String senate) {
        int R=0,D=0;
        int n = senate.length();
        for(int i=0;i<n;++i){
            if(senate.charAt(i)=='R')
                R++;
            else
                D++;
        }
        if(R==0||D==0)
            return D==0?"Radiant":"Dire";
        StringBuilder sb = new StringBuilder(senate);
        while(true){
            n = sb.length();
            String newSenate = sb.toString();
            boolean []killed=new boolean[n];
            int indR=0,indD=0;
            sb.setLength(0);
            for(int i=0;i<n;++i){
                if(killed[i])
                    continue;
                sb.append(newSenate.charAt(i));
                char c = newSenate.charAt(i);
                char oppo= c=='D'?'R':'D';
                int start = oppo=='R'?indR:indD;
                start = Math.max(i+1,start);
                int ind = newSenate.indexOf(oppo,start);
                if(ind>=0 && ind<n)
                    killed[ind]=true;
                if(oppo=='R'){
                    indR = ind+1;
                    R--;
                }else{
                    indD = ind+1;
                    D--;
                }
                if(R==0||D==0)
                    return D==0?"Radiant":"Dire";
            }
        }
    }


    //576 out of boundary path
    public int findPaths(int m, int n, int N, int i, int j) {
        return 0;
    }


    //659. Split Array into Consecutive Subsequences
    public boolean isPossible(int[] nums) {
        int n = nums.length;
        Map<Integer,PriorityQueue<Integer>>map = new HashMap<>();//key is the last number, list is length of list
        for(int x:nums){
            if(x>Integer.MIN_VALUE && map.containsKey(x-1)){
                PriorityQueue<Integer>pq = map.get(x-1);
                int len = pq.poll();
                if(pq.isEmpty())
                    map.remove(x-1);
                PriorityQueue<Integer>pq1 = map.getOrDefault(x,new PriorityQueue<>());
                pq1.offer(len+1);
                map.put(x,pq1);
            }else{
                PriorityQueue<Integer>pq=map.getOrDefault(x,new PriorityQueue<>());
                pq.offer(1);
                map.put(x,pq);
            }
        }

        for(Map.Entry<Integer,PriorityQueue<Integer>>entry:map.entrySet()){
            PriorityQueue<Integer>pq = entry.getValue();
            while(!pq.isEmpty()){
                int top = pq.poll();
                if(top<3)
                    return false;
            }
        }
        return true;
    }

    public boolean isPossibleEasy(int[] nums) {
        Map<Integer, Integer> freq = new HashMap<>(), appendfreq = new HashMap<>();
        for (int i : nums) freq.put(i, freq.getOrDefault(i,0) + 1);
        for (int i : nums) {
            if (freq.get(i) == 0) continue;
            else if (appendfreq.getOrDefault(i,0) > 0) {
                appendfreq.put(i, appendfreq.get(i) - 1);
                appendfreq.put(i+1, appendfreq.getOrDefault(i+1,0) + 1);
            }
            else if (freq.getOrDefault(i+1,0) > 0 && freq.getOrDefault(i+2,0) > 0) {
                freq.put(i+1, freq.get(i+1) - 1);
                freq.put(i+2, freq.get(i+2) - 1);
                appendfreq.put(i+3, appendfreq.getOrDefault(i+3,0) + 1);
            }
            else return false;
            freq.put(i, freq.get(i) - 1);
        }
        return true;
    }



    public void dfs1(List<List<Integer>>res,boolean[]vis,int n,List<Integer>path){
        if(path.size()==n){
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i=1;i<=n;++i){
            if(vis[i]||i==path.size()+1)
                continue;
            vis[i]=true;
            path.add(i);
            dfs1(res,vis,n,path);
            path.remove(path.size()-1);
            vis[i]=false;
        }
    }
    public List<List<Integer>> findDerangement(int n) {
        List<List<Integer>>res = new ArrayList<>();
        boolean []vis = new boolean[n+1];
        dfs1(res,vis,n,new ArrayList<>());
        System.out.println(res.size());
        return res;
    }
}




