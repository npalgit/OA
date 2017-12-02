import commons.Interval;
import commons.TreeNode;
import jdk.nashorn.internal.runtime.regexp.joni.Regex;

import javax.print.DocFlavor;
import java.util.*;

/**
 * Created by tao on 8/20/17.
 */
public class NewQuestions {


    //661 image smoother
    public int[][]imageSmother(int[][]M){
        if(M.length==0||M[0].length==0)
            return M;
        int m = M.length,n=M[0].length;
        int [][]matrix = new int[m][n];
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                int sum =0;
                int cnt=0;
                for(int l=i-1;l<=i+1;++l){
                    for(int r =j-1;r<=j+1;++r){
                        if(l>=0 && l<m && r>=0 && r<n){
                            sum+=M[l][r];
                            cnt++;
                        }
                    }
                }
                matrix[i][j]=sum/cnt;
            }
        }
        return matrix;
    }


    //479 largest palindrome product

    public int largestPalindrome(int n){
        if(n==1)
            return 9;
        int upperBound=(int)Math.pow(10,n)-1,lowerBound = upperBound/10;
        long maxNum = (long)upperBound*(long)upperBound;

        int firstHalf = (int)(maxNum/(long)Math.pow(10,n));

        boolean palindromeFound = false;
        long palidrome = 0;
        while(!palindromeFound){
            palidrome = createPalindrome(firstHalf);
            for(long i = upperBound;upperBound>=lowerBound;--i){
                if(palidrome/i>maxNum||i*i<palidrome){
                    break;
                }
                if(palidrome%i==0){
                    palindromeFound=true;
                    break;
                }
            }
            firstHalf--;
        }
        return (int)(palidrome%1337);
    }

    public long createPalindrome(long num){
        String str = num + new StringBuilder().append(num).reverse().toString();
        return Long.parseLong(str);

    }


    //662 maximum width of binary tree
    //a little complexed
    public void rightView(TreeNode root, List<TreeNode>right, int level){
        if(root==null)
            return;
        if(level>=right.size())
            right.add(root);
        rightView(root.right,right,level+1);
        rightView(root.left,right,level+1);
    }

    public void leftView(TreeNode root,List<TreeNode>left,int level){
        if(root==null)
            return;
        if(level>=left.size())
            left.add(root);
        leftView(root.left,left,level+1);
        leftView(root.right,left,level+1);
    }

    public void dfs(TreeNode root,Map<TreeNode,Integer>map,int code){
        if(root==null)
            return;
        map.put(root,code);
        dfs(root.left,map,2*code);
        dfs(root.right,map,2*code+1);
    }
    public int widthOfBinaryTree(TreeNode root){
        if(root==null)
            return 0;
        int val=0;
        Map<TreeNode,Integer>map=new HashMap<>();
        dfs(root,map,1);
        List<TreeNode>left=new ArrayList<>();
        List<TreeNode>right=new ArrayList<>();
        leftView(root,left,0);
        rightView(root,right,0);
        int n = left.size();
        for(int i=0;i<n;++i){
            int num = map.get(right.get(i))-map.get(left.get(i))+1;
            if(val<num)
                val=num;
        }
        return val;
    }


    //663 equal tree partition
    public int dfs(TreeNode root,Map<TreeNode,Integer>map,Map<Integer,Integer>set){
        if(root==null)
            return 0;
        int left = dfs(root.left,map,set);
        int right = dfs(root.right,map,set);
        map.put(root,left+right+root.val);
        int cnt = set.getOrDefault(left+right,0);
        set.put(left+right+root.val,cnt+1);
        return left+right+root.val;
    }
    public boolean checkEqualTree(TreeNode root){
        //two sum
        if(root==null)
            return false;
        Map<TreeNode,Integer>map=new HashMap<>();
        Map<Integer,Integer>count = new HashMap<>();
        dfs(root,map,count);
        int sum = map.get(root);
        if(sum==0 && count.get(sum)>1)
            return true;
        if(sum!=0 && sum%2==0 && count.containsKey(sum/2))
            return true;
        return false;
    }

    //655 print binary tree
    public int getHeight(TreeNode node){
        if(node==null)
            return 0;
        return 1+Math.max(getHeight(node.left),getHeight(node.right));
    }

    public void dfs(TreeNode root,List<List<String>>res,int level,int index,int offset){
        if(root==null)
            return;
        res.get(level).set(index,String.valueOf(root.val));
        dfs(root.left,res,level+1,index-(1<<(offset-1)),offset-1);
        dfs(root.right,res,level+1,index+(1<<(offset-1)),offset-1);
    }
    public List<List<String>>printTree(TreeNode root){
        int m = getHeight(root);
        int n=(1<<m)-1;
        String []args = new String[n];
        Arrays.fill(args,"");
        List<List<String>>res = new ArrayList<>();
        for(int i=0;i<m;++i){
            res.add(new ArrayList<>());
            res.get(i).addAll(Arrays.asList(args));
        }
        dfs(root,res,0,n/2,m-1);
        return res;
    }

    public String dfs652(TreeNode root,Map<TreeNode,String>map,Map<String,Integer>cnt){
        if(root==null)
            return "";
        String left = dfs652(root.left,map,cnt);
        String right = dfs652(root.right,map,cnt);
        String res = (left.isEmpty()?"l":left)+"@"+root.val+"@"+(right.isEmpty()?"r":right);
        int count = cnt.getOrDefault(res,0);
        cnt.put(res,count+1);
        map.put(root,res);
        return res;
    }
    public List<TreeNode> findDuplicateSubtrees(TreeNode root){
        List<TreeNode>res=new ArrayList<>();
        Map<TreeNode,String>map=new HashMap<>();
        Map<String,Integer>cnt=new HashMap<>();
        dfs652(root,map,cnt);
        Set<String>sets=new HashSet<>();
        for(Map.Entry<TreeNode,String>entry:map.entrySet()){
            String val = entry.getValue();
            if(cnt.getOrDefault(val,0)>1 && !sets.contains(val)){
                sets.add(val);
                res.add(entry.getKey());
            }
        }
        return res;
    }




    //555 split concatenated strings
    public String getBigger(String str){
        char []ss = str.toCharArray();
        boolean larger = true;
        int begin =0,end=ss.length-1;
        while(end>=0){
            if(ss[begin]>ss[end]){
                larger=true;
                break;
            }else if(ss[begin]<ss[end]){
                larger=false;
                break;
            }else{
                begin++;
                end--;
            }
        }
        return larger?str:new StringBuilder(str).reverse().toString();
    }

    public int isBigger(String a,String b){
        char []aa=a.toCharArray();
        char []bb = b.toCharArray();
        int n = aa.length;
        for(int i=0;i<n;++i){
            if(aa[i]>bb[i])
                return 1;
            else if(aa[i]<bb[i])
                return  -1;
        }
        return 0;
    }
    public String splitLoopedString(String[]strs){
        StringBuilder sb = new StringBuilder();
        StringBuilder original = new StringBuilder();
        char maxChar = 'a';
        int n = strs.length;
        for(int i=0;i<n;++i){
            sb.append(getBigger(strs[i]));
            original.append(strs[i]);
            for(int j=0;j<strs[i].length();++j){
                if(maxChar<strs[i].charAt(j))
                    maxChar=strs[i].charAt(j);
            }
        }
        List<Integer>indexes=new ArrayList<>();
        Map<Integer,Interval>map=new HashMap<>();
        int start=0;
        for(int i=0;i<n;++i){
            for(int j=0;j<strs[i].length();++j){
                if(maxChar==strs[i].charAt(j)){
                    indexes.add(start+j);
                    map.put(start+j,new Interval(start,start+strs[i].length()));
                }
            }
            start+=strs[i].length();
        }
        String xx = sb.toString();
        String res = xx;
        String orig = original.toString();
        for(int ind:indexes){
            Interval interval = map.get(ind);
            String changed = orig.substring(interval.start,interval.end);
            String tmp1 = orig.substring(ind,interval.end)+xx.substring(interval.end)+xx.substring(0,interval.start)+orig.substring(interval.start,ind);
            if(isBigger(res,tmp1)<0)
                res=tmp1;
            ind=interval.end-ind-1;
            String reverseChanged = new StringBuilder(changed).reverse().toString();
            String tmp2 = reverseChanged.substring(ind)+xx.substring(interval.end)+xx.substring(0,interval.start)+reverseChanged.substring(0,ind);
            if(isBigger(res,tmp2)<0)
                res=tmp2;
        }
        return res;
    }


    //529 minesweeper
    public char[][]updateBorad(char[][]board,int[]click){
        if(board.length==0||board[0].length==0)
            return board;
        if(board[click[0]][click[1]]=='M'){
            board[click[0]][click[1]]='X';
            return board;
        }
        int m = board.length,n=board[0].length;
        boolean [][]vis=new boolean[m][n];
        Queue<int[]>q=new LinkedList<>();
        q.offer(click);
        int[][]dirs={{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
        vis[click[0]][click[1]]=true;
        while(!q.isEmpty()){
            //judge first;
            int []top = q.poll();
            int cnt=0;
            Queue<int[]>q1=new LinkedList<>();
            for(int []dir:dirs){
                int nx = dir[0]+top[0];
                int ny = dir[1]+top[1];
                if(nx<0||nx>=m||ny<0||ny>=n||vis[nx][ny])
                    continue;
                if(board[nx][ny]=='M'){
                    cnt++;
                }else if(board[nx][ny]=='E'){
                    q1.offer(new int[]{nx,ny});
                }
            }
            if(cnt==0){
                board[top[0]][top[1]]='B';
                while(!q1.isEmpty()){
                    int[]top1 = q1.poll();
                    q.offer(top1);
                    vis[top1[0]][top1[1]]=true;
                }

            }
            else
                board[top[0]][top[1]]=(char)('0'+cnt);
        }
        return board;

    }

    public int maximumSwap(int num) {
        char []ss =String.valueOf(num).toCharArray();
        int n = ss.length;
        boolean isReverse=true;
        int []cnt=new int[10];
        cnt[ss[0]-'0']++;
        for(int i=1;i<n;++i){
            if(ss[i-1]<ss[i]){
                isReverse=false;
            }
            cnt[ss[i]-'0']++;
        }
        if(isReverse)
            return num;
        StringBuilder sb = new StringBuilder();
        for(int i=9;i>=0;--i){
            for(int j=0;j<cnt[i];++j)
                sb.append(i);
        }
        for(int i=0;i<n;++i){
            if(ss[i]!=sb.charAt(i)){
                char c = ss[i];
                ss[i]=sb.charAt(i);
                int j =n-1;
                while(j>=0){
                    if(ss[j]==ss[i])
                        break;
                    j--;
                }
                ss[j]=c;
                //find the last
                break;
            }
        }
        return Integer.parseInt(String.valueOf(ss));

    }

    //625 minimum factorization
    Map<Integer,String>map=new HashMap<>();
    public String smallest(int a){
        if(a<=9){
            return String.valueOf(a);
        }
        if(map.containsKey(a))
            return map.get(a);
        String res="";
        for(int i=2;i<=9;++i){
            if(a%i==0){
                String tmp = ""+i+smallest(a/i);
                if(tmp.indexOf("0")!=-1||tmp.length()>10||Long.parseLong(tmp)>2147483647)
                    continue;
                if(res.isEmpty()||Long.parseLong(tmp)<Long.parseLong(res)){
                    res=tmp;
                }
            }
        }

        String ans =res.isEmpty()?"0":res;
        map.put(a,ans);
        return ans;
    }

    //shopping offers
    public int getMoney(List<Integer>price,List<Integer>needs){
        int n = price.size(),res=0;
        for(int i=0;i<n;++i){
            res+=price.get(i)*needs.get(i);
        }
        return res;
    }
    public int shoppingOffers(List<Integer> price, List<List<Integer>> specials, List<Integer> needs) {
        int res = getMoney(price,needs);
        int j=0;
        for(List<Integer>special:specials){
            List<Integer>copy = new ArrayList<>(needs);
            for(;j<needs.size();++j){
                int diff = copy.get(j)-special.get(j);
                if(diff<0)
                    break;
                copy.set(j,diff);
            }
            if(j==needs.size())
                res=Math.min(res,special.get(j)+shoppingOffers(price,specials,copy));
        }
        return res;
    }


    public int findKthNumer(int n,int k){
        int curr =1;
        k -=1;
        while(k>0){
            int steps = calSteps(n,curr,curr+1);
            if(steps<=k){
                k-=steps;
                curr+=1;
            }else{
                curr *=10;
                k -=1;
            }
        }
        return curr;
    }

    public int calSteps(int n,long n1,long n2){
        int steps=0;
        while(n1<=n){
            steps +=Math.min(n+1,n2)-n1;
            n1*=10;
            n2 *=10;
        }
        return steps;
    }



    public int lengthLongestPath(String input) {
        Stack<Integer>stk = new Stack<>();
        Stack<Integer>cur = new Stack<>();
        String []inputs = input.split("\n");
        int len =0,i=0,n=inputs.length;
        stk.push(-1);
        cur.push(0);
        while(i<n){

            char []ss = inputs[i].toCharArray();
            int ii =0,nn=ss.length;
            while(ii<nn){
                if(ss[ii]!='\t')
                    break;
                ii++;
            }
            while(!stk.isEmpty() && stk.peek()!=ii-1){
                cur.pop();
                stk.pop();
            }
            if(inputs[i].indexOf(".")!=-1){
                len=Math.max((cur.isEmpty()?0:cur.peek())+nn-ii,len);
            }else{
                int val = cur.isEmpty()?0:cur.peek();
                cur.push(val+nn-ii+1);
                stk.push(ii);
            }
            i++;
        }
        return len;
    }


    public boolean isSubsequence(String s, String t) {
        List<Integer>[] idx = new List[256]; // Just for clarity
        for (int i = 0; i < t.length(); i++) {
            if (idx[t.charAt(i)] == null)
                idx[t.charAt(i)] = new ArrayList<>();
            idx[t.charAt(i)].add(i);
        }

        int prev = 0;
        for (int i = 0; i < s.length(); i++) {
            if (idx[s.charAt(i)] == null) return false; // Note: char of S does NOT exist in T causing NPE
            //see this usage at first time
            int j = Collections.binarySearch(idx[s.charAt(i)], prev);
            if (j < 0) j = -j - 1;
            if (j == idx[s.charAt(i)].size()) return false;
            prev = idx[s.charAt(i)].get(j) + 1;
        }
        return true;
    }

    public String decodeString(String s) {
        StringBuilder sb = new StringBuilder();
        if(s.indexOf("[")==-1)
            return s;
        int ind =0,n = s.length(),sum=0;
        char []ss=s.toCharArray();
        while(ind<n){
            if(Character.isLetter(ss[ind])){
                sb.append(ss[ind++]);
            }else if(Character.isDigit(ss[ind])){
                sum=10*sum+(ss[ind++]-'0');
            }else if(ss[ind]=='['){
                int cnt= 0,start=ind+1;
                while(ind<n){
                    if(ss[ind]=='[')
                        cnt++;
                    if(ss[ind]==']')
                        cnt--;
                    if(cnt==0)
                        break;
                    ind++;
                }
                String tmp = decodeString(s.substring(start,ind));
                while(sum-- >0){
                    sb.append(tmp);
                }
                sum=0;
                ind++;
            }
        }
        return sb.toString();
    }

    public boolean valid(int[]data,int i, int cnt){
        int n = data.length;
        for(int j=1;j<=cnt;++j){
            if(i+j>=n||!((data[i+j]&0xff)>=128 && (data[i+j]&0xff)<192))
            return false;
        }
        return true;
    }

    public int findNthDigit(int n) {
        int len = 1;
        long count = 9;
        int start = 1;

        while (n > len * count) {
            n -= len * count;
            len += 1;
            count *= 10;
            start *= 10;
        }

        start += (n - 1) / len;
        String s = Integer.toString(start);
        return Character.getNumericValue(s.charAt((n - 1) % len));
//        int level = 1,exp=1;
//        if(n<=9)
//            return n;
//        while(n>0){
//            int num = level*9*exp;
//            if(n>num){
//                n-=num;
//            }else{
//                break;
//            }
//            exp*=10;
//            level++;
//        }
//        //level;
//        if(n%level==0){
//            return String.valueOf(n/level+exp-1).charAt(level-1)-'0';
//        }else
//            return String.valueOf(n/level+exp).charAt(n%level-1)-'0';

    }



    public double dfs(Map<String,Map<String,Double>>map,Map<String,List<String>>neighbors,String start,String end,Set<String>vis){
        if(vis.contains(start))
            return -1.0;
        if(start.equals(end))
            return 1.0;
        double res = -1.0;
        List<String>nexts = neighbors.getOrDefault(start,new ArrayList<>());
        vis.add(start);
        for(String next:nexts){
            if(!vis.contains(next)){
                double val = dfs(map,neighbors,next,end,vis);
                if(val==-1.0)
                    continue;
                return val * map.get(start).get(next);
            }
        }
        return res;

    }
    public double[] calcEquation(String[][] equations, double[] values, String[][] queries) {
        int n = queries.length,m=equations.length;
        double []res = new double[n];
        Arrays.fill(res,-1.0);
        Map<String,Map<String,Double>>map = new HashMap<>();
        Map<String,List<String>>neighbors = new HashMap<>();
        for(int i=0;i<m;++i){
            String[] equation = equations[i];
            if(!map.containsKey(equation[0]))
                map.put(equation[0],new HashMap<>());
            map.get(equation[0]).put(equation[1],values[i]);
            if(!map.containsKey(equation[1]))
                map.put(equation[1],new HashMap<>());
            map.get(equation[1]).put(equation[0],1.0/values[i]);
            if(!neighbors.containsKey(equation[0]))
                neighbors.put(equation[0],new ArrayList<>());
            neighbors.get(equation[0]).add(equation[1]);

            if(!neighbors.containsKey(equation[1]))
                neighbors.put(equation[1],new ArrayList<>());
            neighbors.get(equation[1]).add(equation[0]);

        }
        for(int i=0;i<n;++i){
            res[i]=dfs(map,neighbors,queries[i][0],queries[i][1],new HashSet<>());
        }
        return res;
    }

    class Node{
        int start;
        int end;
        int isCovered;
        int len;

        public Node(int start,int end,int isCovered,int len){
            this.start=start;
            this.end=end;
            this.isCovered=isCovered;
            this.len=len;
        }
    }
    class Segment{
        int start;
        int end;
        int h;
        int f;
        public Segment(int start,int end,int h,int f){
            this.start=start;
            this.end=end;
            this.h=h;
            this.f=f;
        }
    }

    public int binarySearch(int[]xPositions,int len,int target){
        int begin=0,end=len-1;
        while(begin<end){
            int mid =(end-begin)/2+begin;
            if(xPositions[mid]==target)
                return mid;
            else if(xPositions[mid]<target)
                begin=mid+1;
            else
                end=mid;
        }
        return xPositions[begin]==target?begin:-1;
    }

    void pushup(Node[]nodes,int root,int[]xPositions)
    {
        if (nodes[root].isCovered>0) //非零，已经被整段覆盖
        {
            nodes[root].len = xPositions[nodes[root].end+1] - xPositions[nodes[root].start];
        }
        else if (nodes[root].start == nodes[root].end) //这是一个点而不是线段
        {
            nodes[root].len = 0;
        }
        else //是一条没有整个区间被覆盖的线段，合并左右子的信息
        {
            nodes[root].len = nodes[root*2].len + nodes[root*2+1].len;
        }
    }
    void update(Node[]nodes,int root,int l,int r,int f,int[]xPositions)//这里深刻体会为什么令下边为1，上边-1
    {                                   //下边插入边，上边删除边
        if (nodes[root].start >= l &&nodes[root].end <= r)
        {
            nodes[root].isCovered+= f;
            pushup(nodes,root,xPositions);//更新区间被覆盖de总长度
            return;
        }
        int mid = (nodes[root].end-nodes[root].start)/2+nodes[root].start;
        if (r <= mid) update(nodes,2*root,l,r,f,xPositions);
        else if (l > mid) update(nodes,2*root+1,l,r,f,xPositions);
        else
        {
            update(nodes,2*root,l,mid,f,xPositions);
            update(nodes,2*root+1,mid,r,f,xPositions);
        }
        pushup(nodes,root,xPositions);
    }

    void build(Node[]nodes,int root,int l,int r)
    {
        nodes[root]=new Node(l,r,0,0);
        if (l == r) return;
        int mid =(nodes[root].end-nodes[root].start)/2+nodes[root].start;
        build(nodes,2*root,l,mid);
        build(nodes,2*root+1,mid+1,r);
    }

    public boolean isRectangleCover(int[][] rectangles) {
        int n = rectangles.length;
        Node []nodes = new Node[8*n];
        List<Segment>segments = new ArrayList<>();
        int []xPositions = new int[2*n];
        int ind =0;
        int minX=Integer.MAX_VALUE,minY=Integer.MAX_VALUE,maxX=Integer.MIN_VALUE,maxY=Integer.MIN_VALUE;
        int totalArea=0;
        for(int[] rectangle:rectangles){
            minX=Math.min(rectangle[0],minX);
            minY=Math.min(rectangle[1],minY);
            maxX=Math.max(maxX,rectangle[2]);
            maxY=Math.max(maxY,rectangle[3]);
            segments.add(new Segment(rectangle[0],rectangle[2],rectangle[1],1));
            segments.add(new Segment(rectangle[0],rectangle[2],rectangle[3],-1));
            xPositions[ind++]=rectangle[0];
            xPositions[ind++]=rectangle[2];
            totalArea+=(rectangle[2]-rectangle[0])*(rectangle[3]-rectangle[1]);
        }
        if(totalArea!=(maxY-minY)*(maxX-minX))
            return false;
        Arrays.sort(xPositions);
        Collections.sort(segments, new Comparator<Segment>() {
            @Override
            public int compare(Segment o1, Segment o2) {
                return o1.h-o2.h;
            }
        });
        //delete the duplicates
        ind =1;
        for (int i = 1;i < 2*n;++i)
        {
            if (xPositions[i] != xPositions[i-1]) //去重
            {
                xPositions[ind++] = xPositions[i];
            }
        }
        int sum =0;
        build(nodes,1,0,ind-1);
        for(int i=0;i<2*n-1;++i){
            int l = binarySearch(xPositions,ind,segments.get(i).start);
            int r = binarySearch(xPositions,ind,segments.get(i).end)-1;
            if(l<=r){
                update(nodes,1,l,r,segments.get(i).f,xPositions);
                sum+=(segments.get(i+1).h-segments.get(i).h)*(nodes[1].len);
            }
        }
        return sum==totalArea;
    }

    class Tuple{
        int len;
        int cnt;
        public Tuple(int len,int cnt){
            this.len=len;
            this.cnt=cnt;
        }
    }
    public int findNumberOfLIS(int[] nums) {
        int n = nums.length,maxLen=0,leng=0;
        Tuple []dp=new Tuple[n];
        for(int i=0;i<n;++i){
            dp[i]=new Tuple(1,1);
            for(int j=i-1;j>=0;--j){
                if(nums[i]>nums[j] && dp[i].len<=dp[j].len+1){
                    if(dp[i].len<dp[j].len+1){
                        dp[i].cnt=dp[j].cnt;
                    }else{
                        dp[i].cnt+=dp[j].cnt;
                    }
                    dp[i].len=dp[j].len+1;
                }
            }
            if(maxLen<=dp[i].len){
                if(maxLen==dp[i].len)
                    leng+=dp[i].cnt;
                else
                    leng=dp[i].cnt;
                maxLen=dp[i].len;
            }
        }
        return leng;
    }


    public boolean isValid(String s){
        int cnt=0,n=s.length();
        if(n%2!=0)
            return false;
        for(int i=0;i<n;++i){
            if(s.charAt(i)=='(')
                cnt++;
            else if(s.charAt(i)==')')
                cnt--;
            if(cnt<0)
                return false;
        }
        return cnt==0;
    }
    Map<String,Boolean>map1=new HashMap<>();
    //change to dp,
    public boolean checkValidString(String s) {
        if(map1.containsKey(s))
            return map1.get(s);
        if(s.isEmpty()||s.indexOf("*")<0){
            boolean end =isValid(s);
            map1.put(s,end);
            return end;
        }
        int n=s.length();
        int cnt=0;
        char []ss=s.toCharArray();
        for(int i=0;i<n;++i){
            if(ss[i]=='(')
                cnt++;
            else if(ss[i]==')')
                cnt--;
            else{
                ss[i]='(';
                if(checkValidString(String.valueOf(ss))){
                    map1.put(s,true);
                    return true;
                }
                ss[i]=')';
                if(cnt-1>=0 && checkValidString(String.valueOf(ss))){
                    map1.put(s,true);
                    return true;
                }
                if(checkValidString(s.substring(0,i)+s.substring(i+1))){
                    map1.put(s,true);
                    return true;
                }

            }
            if(cnt<0){
                map1.put(s,false);
                return false;
            }
        }
        map1.put(s,false);
        return false;
    }

    public boolean checkValidStringBetter(String s) {
        int n=s.length();
        int []dp=new int[n+1];
        boolean []res = new boolean[n+1];
        res[0]=true;
        char []ss = s.toCharArray();
        for(int i=2;i<=n;++i){
            if(ss[i-1]==')'||ss[i-1]=='*'){
                if(ss[i-1]=='*'){
                    res[i]|=res[i-1];
                    dp[i]=Math.max(dp[i],dp[i-1]);
                }
                if(ss[i-2]=='('||ss[i-2]=='*'){
                    dp[i]=Math.max(dp[i],2+dp[i-2]);
                    res[i]|=res[i-2];
                }else{
                    int ind =i-dp[i-1]-1;
                    if(ind>=1 && ss[ind-1]=='('){
                        res[i]=res[i]|(res[i-1] && res[ind-1]);
                        dp[i]=Math.max(dp[i],dp[i-1]+2+dp[ind-1]);
                    }
                }
            }
        }
        return res[n];
    }

    class Point{
        int step;
        int point;
        public Point(int step,int point){
            this.point = point;
            this.step = step;
        }
    }
    public boolean canCross(int[] stones) {
        int n = stones.length;
        Set<Integer>rockes = new HashSet<>();
        for(int x:stones){
            rockes.add(x);
        }
        Queue<Point>q=new LinkedList<>();
        q.offer(new Point(1,1));
        while(!q.isEmpty()){
            Point top = q.poll();
            if(top.point==stones[n-1])
                return true;
            if(top.step>1 && rockes.contains(top.step-1+top.point)){
                q.offer(new Point(top.step-1,top.step-1+top.point));
            }

            if(rockes.contains(top.step+top.point)){
                q.offer(new Point(top.step,top.step+top.point));
            }

            if(rockes.contains(top.step+1+top.point)){
                q.offer(new Point(top.step+1,top.step+top.point+1));
            }

        }
        return false;
    }

    public boolean doable(int[]nums,int sum,int cnt){
        int num = 0, n = nums.length,sum1=0;
        for(int i=0;i<n;++i){
            sum1+=nums[i];
            if(sum1>sum){
                sum1=nums[i];
                num++;
            }
        }
        return num<=cnt-1;
    }
    public int splitArray(int[] nums, int m) {
        int n = nums.length;
        int maxVal=Integer.MIN_VALUE,sum=0;
        for(int i=0;i<n;++i){
            maxVal=Math.max(maxVal,nums[i]);
            sum+=nums[i];
        }
        if(m==1)
            return sum;
        int begin = maxVal,end=sum;
        while(begin<=end){
            int mid = (end-begin)/2+begin;
            if(doable(nums,mid,m))
                end=mid-1;
            else
                begin=mid+1;
        }
        return begin;
    }


    public void dfs(List<String>res,int cnt,int ind,String word,String path,List<String>[]args,boolean stored){
        if(ind==word.length()){
            String ele = path+(cnt!=0?""+cnt:"");
            if(stored)
                res.add(ele);
            else
                args[getLength(ele)].add(ele);
            return;
        }
        //this character is continue to be numeric
        dfs(res,cnt+1,ind+1,word,path,args,stored);

        //break
        dfs(res,0,ind+1,word,path+(cnt!=0?cnt+"":"")+word.charAt(ind),args,stored);
    }
    public List<String>getAbbr(String word){
        List<String>res = new ArrayList<>();
        List<String>[]args = new ArrayList[word.length()+1];
        dfs(res,0,0,word,"",args,true);
        return res;
    }
    public List<String>[] getAbbrArray(String word){
        List<String>res = new ArrayList<>();
        List<String>[]args = new ArrayList[word.length()+1];
        for(int i=0;i<args.length;++i)
            args[i]=new ArrayList<>();
        dfs(res,0,0,word,"",args,false);
        return args;
    }
    public int getLength(String s){
        int cnt =0,n=s.length(),i=0;
        while(i<n){
            if(Character.isDigit(s.charAt(i))){
                while(i<n && Character.isDigit(s.charAt(i)))
                    i++;
            }else{
                i++;
            }
            cnt++;
        }
        return cnt;
    }
    public String minAbbreviation(String target, String[] dictionary) {
        Set<String>words = new HashSet<>();
        for(String str:dictionary)
            words.addAll(getAbbr(str));
        List<String>[]res = getAbbrArray(target);
         int n=target.length();
         for(int i=1;i<=n;++i){
             for(String str:res[i])
                 if(!words.contains(str))
                     return str;
         }
        return "";
    }

    public int numberOfArithmeticSlices(int[] A) {
        int n = A.length;
        Map<Long,Map<Long,Integer>>dp=new HashMap<>();//dp[i][j] is the number of subsequence ends with i, len is j
        Map<Long,List<Integer>>map =new HashMap<>();//number and its index;
        Map<Integer,Integer>number = new HashMap<>();//duplicate numebr;
        for(int i=0;i<n;++i){
            if(i>=2){
                Set<Integer>vis = new HashSet<>();
                for(int j =i-1;j>=1;--j){
                    long before  = 2*((long)A[j])-(long)A[i];
                    if(dp.containsKey(before)){
                            for(int index:map.get(before)){
                                if(index<j){
                                    long gap = (long)A[j]- before;
                                    if(gap!=0){
                                        int val =0;
                                        if(!vis.contains(A[j])){
                                            val = dp.get((long)A[j]).getOrDefault(gap,0);
                                            vis.add(A[j]);
                                        }
                                        if(!dp.containsKey((long)A[i]))
                                            dp.put((long)A[i],new HashMap<>());
                                        int valll = dp.get((long)A[i]).getOrDefault(gap,0);
                                        dp.get((long)A[i]).put(gap,val+1+valll);
                                    }
                                }
                            }
                    }
                }
            }
            if(!dp.containsKey((long)A[i]))
                dp.put((long)A[i],new HashMap<>());
            if(!map.containsKey((long)A[i]))
                map.put((long)A[i],new ArrayList<>());
            map.get((long)A[i]).add(i);
            int vall = number.getOrDefault(A[i],0);
            number.put(A[i],vall+1);
        }
        int res =0;
        for(Map.Entry<Long,Map<Long,Integer>>en:dp.entrySet()){

            Map<Long,Integer>it = en.getValue();
            if(it!=null){
                for(Map.Entry<Long,Integer>entry:it.entrySet()){
                    res += entry.getValue();
                }
            }
        }

        for(Map.Entry<Integer,Integer>entry:number.entrySet()){
            int val = entry.getValue();
            if(val>=3)
                res += getNum(val);
        }
        return res;
    }

    public int getNum(int val){
        return (1<<val)-1-val*(val+1)/2;
    }







    public void minus(int[]cnt,String s,int count){
        int n = s.length();
        while(count-- >0){
        for(int i=0;i<n;++i)
            cnt[s.charAt(i)-'a']--;
        }
    }
    public void add(StringBuilder sb, int num,int cnt){
        while(cnt-- >0)
            sb.append(num);
    }
    public String originalDigits(String s) {
        //four zero eight six two
        // u    z    g   x   w


        //then
        //three one seven five
        // t   o  s  f

        //nine

        char []ss = s.toCharArray();
        int []cnt = new int[26];
        for(char c:ss)
            cnt[c-'a']++;
        StringBuilder sb = new StringBuilder();
        int count = cnt['u'-'a'];
        add(sb,4,count);
        minus(cnt,"four",count);
        count = cnt['z'-'a'];
        add(sb,0,count);
        minus(cnt,"zero",count);

        count = cnt['g'-'a'];
        add(sb,8,count);
        minus(cnt,"eight",count);

        count = cnt['x'-'a'];
        add(sb,6,count);
        minus(cnt,"six",count);

        count = cnt['w'-'a'];
        add(sb,2,count);
        minus(cnt,"two",count);

        count = cnt['t'-'a'];
        add(sb,3,count);
        minus(cnt,"three",count);

        count = cnt['o'-'a'];
        add(sb,1,count);
        minus(cnt,"one",count);

        count = cnt['s'-'a'];
        add(sb,7,count);
        minus(cnt,"seven",count);

        count = cnt['f'-'a'];
        add(sb,5,count);
        minus(cnt,"five",count);


        count = cnt['e'-'a'];
        add(sb,9,count);
        minus(cnt,"nine",count);

        char []tt = sb.toString().toCharArray();
        Arrays.sort(tt);
        return String.valueOf(tt);
    }


    public List<Integer> findAnagrams(String s, String p) {
        //two pointers 搞一搞
        int []cnt = new int[26];
        char []pp= p.toCharArray();
        char []ss = s.toCharArray();
        for(char c:pp)
            cnt[c-'a']++;
        int d = pp.length, start =0 , n = s.length(),end=0,m=pp.length;
        List<Integer>res = new ArrayList<>();
        while(end<n){
            if(cnt[ss[end++]-'a']-- >0)
                d--;
            if(d==0)
                res.add(start);
            if(end>=start+m && cnt[ss[start++]-'a']++ >=0)
                d++;
        }
        return res;
    }



    public int findMinArrowShots(int[][] points) {
        int n = points.length;
        if(n<=1)
            return n;
        Arrays.sort(points,new Comparator<int[]>(){
            public int compare(int[]a1,int[]a2){
                return a1[0]-a2[0];
            }
        });
        int cnt =1,end=points[0][1];
        for(int i=1;i<n;++i){
            if(points[i][0]>end){
                cnt++;
                end = points[i][1];
            }else{
                end = Math.min(end,points[i][1]);
            }
        }
        return cnt;

    }


    //[[0,1,5],[2,3,1],[2,0,1],[4,0,2]]
    //2,2,2,  -1,-5
    //The question can be transferred to a 3-partition problem, which is NP-Complete.
    public int minTransfers(int[][] transactions) {
        Map<Integer,Long>map= new HashMap<>();
        for(int[]tran:transactions){
            long val1 = map.getOrDefault(tran[0],0L);
            map.put(tran[0],val1+tran[2]);
            long val2 = map.getOrDefault(tran[1],0L);
            map.put(tran[1],val2-tran[2]);
        }
        List<Long> list = new ArrayList();
        for(long val: map.values()){
            if(val != 0) list.add(val);
        }
        Long[] debts = new Long[list.size()];
        debts = list.toArray(debts);
        return deal(debts, 0 , 0);
    }

    public int deal(Long[] debts, int pos, int count){
        while(pos < debts.length && debts[pos] == 0) pos++;
        if (pos >= debts.length) {
            return count;
        }
        int res = Integer.MAX_VALUE;
        for(int i = pos + 1; i < debts.length; i++){
            if(debts[pos] * debts[i] < 0){
                debts[i] += debts[pos];
                res = Math.min(res, deal(debts, pos + 1, count + 1));
                debts[i] = debts[i] - debts[pos];
            }
        }
        return res;
    }

    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        TreeMap<Integer,Integer>tree = new TreeMap<>();
        for(int x:s){
            int val = tree.getOrDefault(x,0);
            tree.put(x,val+1);
        }
        int cnt =0;
        for(int x:g){
            Map.Entry<Integer,Integer>entry = tree.ceilingEntry(x);
            if(entry!=null && entry.getValue()>=1){
                cnt++;
                if(entry.getValue()==1)
                    tree.remove(entry.getKey());
                else
                    tree.put(entry.getKey(),entry.getValue()-1);
            }
        }
        return cnt;
    }

    public boolean find132pattern(int[] nums) {
        Stack<Integer> stk=new Stack<>();
        int n=nums.length,s3=Integer.MIN_VALUE;
        for(int i=n-1;i>=0;--i){
            if(nums[i]<s3)
                return true;
            while(!stk.isEmpty() && stk.peek()<nums[i]){
                s3=stk.pop();
            }
            stk.push(nums[i]);
        }
        return false;
    }

    //tle
    public void dealwith(Map<Character,Set<Integer>>map,char[]pp,int start,int end){
        int len = end-start+1;
        for(int i=start;i<=end;++i){
            if(!map.containsKey(pp[i]))
                map.put(pp[i],new HashSet<>());
            for(int j=1;j<=len;++j)
                map.get(pp[i]).add(j);
            len--;
        }
    }
    public int findSubstringInWraproundString(String p) {
        Map<Character,Set<Integer>>map=new HashMap<>();//start character, len
        int n = p.length();
        char []pp = p.toCharArray();
        int i=1,start=0;
        while(i<=n){
            if(i<n && (pp[i]-pp[i-1]+26)%26==1){
            }else{
                dealwith(map,pp,start,i-1);
                start=i;
            }
            i++;
        }
        int cnt = 0;
        for(Map.Entry<Character,Set<Integer>>entry:map.entrySet()){
            cnt+=entry.getValue().size();
        }
        return cnt;
    }

    //26 个字母
    public int findSubstringInWraproundStringDP(String p) {
         int []dp=new int[26];
         int cnt =0,n=p.length();
         int []count=new int[26];
         for(int i=0;i<n;++i)
             count[p.charAt(i)-'a']++;
         for(int i=0;i<26;++i){
             if(count[i]==0)
                 continue;
             int begin =1,end=n;
             StringBuilder sb = new StringBuilder();
             while(begin<=end){
                 sb.setLength(0);
                 int mid=(end-begin)/2+begin;
                 for(int j=0;j<mid;++j)
                     sb.append((char)((i+j)%26+'a'));
                 if(p.indexOf(sb.toString())!=-1)
                     begin=mid+1;
                 else
                     end=mid-1;
             }
             int nn = sb.length();
             if(nn<begin){
                 char cc = (char)('a'+(sb.charAt(nn-1)+1-'a')%26);
                 sb.append(cc);
             }
             dp[i]=p.indexOf(sb.toString())!=-1?begin:begin-1;
         }

         for(int i=0;i<26;++i){
             cnt += dp[i];
         }
         return cnt;
    }

    public int magicalString(int n) {
        StringBuilder sb = new StringBuilder("122");
        StringBuilder abbr = new StringBuilder("122");
        int cnt =1,j=2;
        while(sb.length()<n){
            int m = sb.length();
            if(abbr.charAt(j)=='1'){
                if(sb.charAt(m-1)=='2'){
                    cnt++;
                    sb.append('1');
                }else
                    sb.append('2');
            }else{
                if(sb.charAt(m-1)=='1')
                    sb.append('2').append('2');
                else{
                    cnt+=2;
                    sb.append('1').append('1');
                }
            }
            abbr.append(sb.charAt(++j));
        }
        System.out.println(sb.toString());
        return cnt;
    }


    //2001:0db8:85a3:0000:0000:8a2e:0370:7334
    //"2001:0db8:85a3:0:0:8A2E::0370:7334:"
    //非常糟糕的一道题
    //regular expression 一招致命
    public String validIPAddress(String IP) {
        String ipv4 = "[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}";
        String ipv6 = "[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4}";
        if(IP.matches(ipv4)){
            String[]args = IP.split("\\.");
            for(int i=0;i<4;++i){
                int val = Integer.parseInt(args[i]);
                if(val<0||val>255||args[i].length()!=1 && args[i].charAt(0)=='0')
                    return "Neither";
            }
            return "IPv4";
        }else if(IP.matches(ipv6)){
            return "IPv6";
        }
        return "Neither";
    }
}
