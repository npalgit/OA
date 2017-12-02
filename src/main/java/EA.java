import commons.Interval;
import commons.TreeNode;

import java.util.*;

public class EA {

    public List<Integer>getSquare(List<Integer>nums){
        int n = nums.size();
        //find the first non-negative number;
        int i=0;
        List<Integer>ans = new ArrayList<>();
        while(i<n){
            if(nums.get(i)>=0)
                break;
            i++;
        }
        int j=i-1;
        while(i<n && j>=0){
            int val1= nums.get(i)*nums.get(i);
            int val2= nums.get(j)*nums.get(j);
            if(val1>=val2){
                ans.add(val2);
                j--;
            }else{
                ans.add(val1);
                i++;
            }
        }
        while (i<n){
            ans.add(nums.get(i)*nums.get(i));
            i++;
        }
        while(j>=0){
            ans.add(nums.get(j)*nums.get(j));
            j--;
        }
        return ans;
    }

    public int longestPalindromeSubseq(String s) {
        int n=s.length();
        int [][]dp=new int[n][n];
        for(int i=n-1;i>=0;--i){
            dp[i][i]=1;
            for(int j=i+1;j<n;++j){
                if(s.charAt(i)==s.charAt(j)){
                    dp[i][j]=dp[i+1][j-1]+2;
                }else{
                    dp[i][j]=Math.max(dp[i+1][j],dp[i][j-1]);
                }
            }
        }
        return dp[0][n-1];
    }


    public void levelOrder(TreeNode root){
        if(root==null)
            return;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while(!q.isEmpty()){
            TreeNode top = q.poll();
            System.out.println(top.val);
            if(top.left!=null)
                q.offer(top.left);
            if(top.right!=null)
                q.offer(top.right);
        }
    }

    public void dfs(TreeNode root,List<List<Integer>>ans,int level){
        if(root==null)
            return;
        if(level>=ans.size())
            ans.add(new ArrayList<>());
        ans.get(level).add(root.val);
        dfs(root.left,ans,level+1);
        dfs(root.right,ans,level+1);
    }
    public List<List<Integer>>levelOrderDFS(TreeNode root){
        List<List<Integer>>ans = new ArrayList<>();
        if(root==null)
            return ans;
        dfs(root,ans,0);
        return ans;
    }
    //stack
    class Tuple{
        public TreeNode node;
        public int level;
        public Tuple(TreeNode node,int level){
            this.node = node;
            this.level = level;
        }
    }
    public List<List<Integer>>levelOrderStack(TreeNode root){
        List<List<Integer>>ans = new ArrayList<>();
        if(root==null)
            return ans;
        Stack<Tuple>stk = new Stack<>();
        stk.push(new Tuple(root,0));
        while(!stk.isEmpty()){
            Tuple top = stk.pop();
            if(top.level>=ans.size())
                ans.add(new ArrayList<>());
            ans.get(top.level).add(top.node.val);
            if(top.node.right!=null)
                stk.push(new Tuple(top.node.right,top.level+1));
            if(top.node.left!=null)
                stk.push(new Tuple(top.node.left,top.level+1));
        }
        return ans;
    }

    public void swap(StringBuilder str,int i,int j){
        char c = str.charAt(i);
        str.setCharAt(i,str.charAt(j));
        str.setCharAt(j,c);
    }
    public void dfs(List<String>ans,int ind,StringBuilder str){
        if(ind==str.length()){
            ans.add(str.toString());
            return;
        }
        //Set<Integer>appeared = new HashSet<>();
        for(int i=ind;i<str.length();++i){
            //if(appeared.add(nums[i])){
            swap(str,i,ind);
            dfs(ans,ind+1,str);
            swap(str,i,ind);
        }
    }
    public List<String>allPermutations(String str){
        List<String>ans = new ArrayList<>();
        if(str.isEmpty())
            return ans;
        dfs(ans,0,new StringBuilder(str));
        return ans;
    }

    //88 merge two sorted array
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int ind = m+n-1;
        m--;
        n--;
        while(m>=0 && n>=0){
            if(nums1[m]>=nums2[n]){
                nums1[ind--]=nums1[m--];
            }else
                nums1[ind--]=nums2[n--];
        }
        while(n>=0)
            nums1[ind--]=nums2[n--];
    }

    public int cloestValueBSTIterative(TreeNode root,double target){
        //assume there is always one
        int res=root.val;
        TreeNode cur=root;
        while(cur!=null){
            if(Math.abs(res*1.0-target)>Math.abs(cur.val*1.0-target))
                res=cur.val;
            if(cur.val>target)
                cur=cur.left;
            else if(cur.val<target)
                cur=cur.right;
            else
                break;
        }
        return res;
    }

    ///dfs way
    int ans=0;
    public void dfs(TreeNode root,double target){
        if(root==null)
            return;
        if(Math.abs(1.0*ans-target)>Math.abs(1.0*root.val-target))
            ans=root.val;
        if(root.val>target)
            dfs(root.left,target);
        else if(root.val<target)
            dfs(root.right,target);

    }
    public int cloestValueBST(TreeNode root,double target){
        ans=root.val;
        dfs(root,target);
        return ans;
    }

    //一道maximum depth of binary tree的coding
    public int maxDepth(TreeNode root){
        if(root==null)
            return 0;
        return 1+Math.max(maxDepth(root.left),maxDepth(root.right));
    }
    //level order 也是可以的

    //207 course schedule
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        //queue
        Queue<Integer>q = new LinkedList<>();
        Map<Integer,List<Integer>>map = new HashMap<>();
        int []indegree = new int[numCourses];
        for(int []edge:prerequisites){
            indegree[edge[0]]++;
            if(!map.containsKey(edge[1])){
                map.put(edge[1],new ArrayList<>());
            }
            map.get(edge[1]).add(edge[0]);
        }
        for(int i=0;i<numCourses;++i)
            if(indegree[i]==0)
                q.offer(i);
        int cnt=0;
        while(!q.isEmpty()){
            int top = q.poll();
            cnt++;
            List<Integer>adj = map.getOrDefault(top,new ArrayList<>());
            for(int x:adj){
                --indegree[x];
                if(indegree[x]==0)
                    q.offer(x);
            }
        }
        return cnt==numCourses;
    }

    public boolean hasCycle(Map<Integer,List<Integer>>map,int start,boolean[]onLoop,boolean[]vis){
        if(vis[start])
            return false;
        if(onLoop[start])
            return true;
        onLoop[start]=true;
        List<Integer>adj = map.getOrDefault(start,new ArrayList<>());
        for(int x:adj){
            if(hasCycle(map,x,onLoop,vis))
                return true;
        }
        vis[start]=true;//涨了见识，就是说当所有的子节点都访问完后才开始标记为0
        // stk.push(start);这是区别，倒叙就可以了
        return false;
    }
    public boolean canFinishDFS(int numCourses,int[][]prerequisites){
        Map<Integer,List<Integer>>map = new HashMap<>();
        for(int []edge:prerequisites){
            if(!map.containsKey(edge[1])){
                map.put(edge[1],new ArrayList<>());
            }
            map.get(edge[1]).add(edge[0]);
        }
        boolean []vis = new boolean[numCourses];
        for(int i=0;i<numCourses;++i){
            if(!vis[i]){
                if(hasCycle(map,i,new boolean[numCourses],vis)){
                    return false;
                }
            }
        }
        return true;
    }





    //632 smallest range
    public int[] smallestRange(List<List<Integer>> nums) {
        int minx=0, miny= Integer.MAX_VALUE, max_i=Integer.MIN_VALUE, min_i=0;
        int n = nums.size();
        int []next= new int[n];
        PriorityQueue<Integer>pq = new PriorityQueue<>(new Comparator<Integer>() {
            @Override
            public int compare(Integer i, Integer j) {
                return nums.get(i).get(next[i])-nums.get(j).get(next[j]);
            }
        });
        for(int i=0;i<n;++i){
            max_i=Math.max(max_i,nums.get(i).get(0));
            pq.offer(i);
        }
        boolean flag = true;
        while(!pq.isEmpty() && flag){
            min_i = pq.poll();
            if(miny-minx>max_i-nums.get(min_i).get(next[min_i])){
                miny = max_i;
                minx = nums.get(min_i).get(next[min_i]);
            }
            if(next[min_i]==nums.get(min_i).size()-1){
                flag= false;
                break;
            }
            next[min_i]++;
            pq.offer(min_i);
            max_i = Math.max(max_i,nums.get(min_i).get(next[min_i]));
        }
        return new int[]{minx,miny};
    }


    //interval
    //heap search in the project

    //interval
    //然后遍历所有task，dp[task.end] = dp[0..(task.start - 1)] + 1，记录其中最大值，国人大哥觉得make sense。但是如果时间是float的或者interval跨度很大就不能这么做了，不知道有没有更好的解法


    //Maximum subarray sum modulo m
    public void maxSubarrayModM(int []nums,int k){
        int prefix = 0, n = nums.length, maxVal = 0;
        TreeSet<Integer>set = new TreeSet<>();
        set.add(0);
        for(int i=0;i<n;++i){
            prefix = (prefix+nums[i])%k;
            maxVal = Math.max(prefix,maxVal);
            Integer low = set.ceiling(prefix+1);
            if(low!=null){
                maxVal=Math.max(maxVal,prefix-low+k);
            }
            set.add(prefix);
        }
        System.out.println(maxVal);
    }
    public static int maxTask(Interval[] intervals, Interval total) {
        int res = 0;
        // dp[i] - until ith interval, maximum tasks
        List<Integer> dp = new ArrayList<>();
        Arrays.sort(intervals, new Comparator<Interval>() {
            @Override
            public int compare(Interval a, Interval b) {
                if (a.start != b.start)     return a.start - b.start;
                else    return a.end - b.end;
            }
        });
        dp.add(1);
        for (int i = 1; i < intervals.length && intervals[i].end <= total.end; i++) {
            dp.add(0);
            for (int j = 0; j < i; j++) {
                if (intervals[j].end <= intervals[i].start) {
                    if (dp.get(j) + 1 > dp.get(i)) {
                        dp.set(i, dp.get(j) + 1);
                        res = Math.max(res, dp.get(i));
                    }
                }
            }
        }
        return res;
    }



    /*
    int find_max(vector<pair<int, int>>& vec, pair<int, int>& bar){
        vector<int> res;
        int first = bar.first;
        int second = bar.second;
        for(int i = 0; i < vec.size(); i++){
                if(vec[i].first >= first && vec[i].second <= second){.鏈枃鍘熷垱鑷�1point3acres璁哄潧
                        res.push_back(vec[i].first);
                        res.push_back(vec[i].second * (-1));
                }
        }
//        cout << res.size() << endl;
        sort(res.begin(), res.end(), comparator);
        int Max = INT_MIN;
        int count = 0;
        for(int i = 0; i < res.size(); i++){
                if(res[i] > 0){
                        count += 1;
                        Max = max(Max, count); 鏉ユ簮涓€浜�.涓夊垎鍦拌鍧�.
                }
                else{
                        count--;
                }
        }
        return res.size() / 2 - Max + 1;
}
     */


    /*
    public class Employee {
    int        employeeId;
    String     name;
    Department dept;

    // other methods would be in here

    @Override
    public int hashCode() {
        int hash = 1;
        hash = hash * 17 + employeeId;
        hash = hash * 31 + name.hashCode();
        hash = hash * 13 + (dept == null ? 0 : dept.hashCode());
        return hash;
    }
}
hashCode() method, which digests the data stored in an instance of the class into a single hash value (a 32-bit signed integer)
     */
    public static void main(String[]args){
        EA ea = new EA();
        List<Integer>ans = new ArrayList<>(Arrays.asList(-4, -2, 1, 3, 5));
        //System.out.println(ea.getSquare(ans));
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left =new TreeNode(4);
        root.left.right = new TreeNode(5);
        root.right.left =new TreeNode(6);
        root.right.right =new TreeNode(7);
        //System.out.println(ea.levelOrderDFS(root));
        //System.out.println(ea.levelOrderStack(root));


        //System.out.println(ea.allPermutations("ABC"));

        int [][]prequisites={{0,1},{0,2},{1,2}};
        //System.out.println(ea.canFinishDFS(3,prequisites));
        Interval[]intervals=new Interval[3];
        intervals[0]=new Interval(1,50);
        intervals[1] = new Interval(51,100);
        intervals[2] = new Interval(1,99);

        int []nums = {3,3,9,9,5};
        ea.maxSubarrayModM(nums,7);
        //System.out.println(ea.getLargestNumberOfTask(intervals,new Interval(1,100)));
        /*

        http://www.1point3acres.com/bbs/thread-145307-1-1.html
        给一个总的time interval，比如[1, 100]，再给一些tasks的time interval，比如[1, 50], [51, 100], [1, 99]，
        每次只能连续执行一个task，然后问最多能在给定的总的time interval里执行多少个task
        好像0-1 背包



        data team
        首先是问了一下简历里的内容，问你实习的内容，其中最challenging的是啥，问的好仔细... Then 说说你做过的最想向他介绍的project，
        讲讲里面的实现，jiay
        其中有用到NoSQL，说说你存储的形式。然后说说你写过的一个MR问题，把思路给他讲讲~

        写一个多叉树的level traversal，自己定义和构造节点，然后写算法吧~用了BFS。然后他说如果用DFS呢，不许用递归，讲了一下思路就OK。stack

        开始问Java的问题，各种各样的问题（师兄求放过...），问Java Servlet和MVC的具体实现。

        我记得我当时也是只面了一道题目 但是前面花了十几分钟讨论map reduce的知识点。但是过了两个小时就收到了二面的通知了


        一面：半小时简历+Java问题： 讲了最近的proect，问了 interface能不能extends多个interface，hashcode是什么，
        hashmap的实现，什么时候会用到hashmap，还有几个问题记不住了；
        然后做题：给一个string，求里面的所有字符的所有permutation. more info on 1point3acres.com
        二面：大概10多分钟简历+Java问题：讲下project里最有挑战的问题，HashMap的实现；然后大概二十分钟两道题:
        Closest Binary Search Tree Value 和 Merge Two Sorted Array；最后问了两个问题结束


        一道maximum depth of binary tree的coding，我代码写出来了在eclipse上测了也过了，完后问我general java questions, e.g.:
        你希望java增加什么feature？完后问我找intersection of two arrays，
        我说用hashtable, time: O(n), space: O(n)，完后问我left join跟inner join的区别。我都答出来了。

        图里找环..

        第一题：一个boolean array，里面是good bad的flag，如果bad发生，他之后就都是bad. 比如good, good, bad, bad,.....,
        bad。求第一个bad。很简单，binary search.
第二题：就问问，没写代码。在binary search tree里面找topk。我感觉我答得不好，好像复杂度就是O(n)吧。按照inorder-traverse. .
第三题：算是第二题的扩展，写binary tree inorder traverse。当时不知道为什么已经心凉了，连recursive solution都写不好。
然后那家伙就说好了好了，有什么问题要问吗？我就放弃的问，听说EA压力很大啊。他回答：没有的事，我觉得挺轻松的，work load非常合理。草草的就结束了。
感觉超级烂的一次。我接着就给recruiter发了个信说自己发挥的不好，能不能再给一次机会。结果那个recruiter就回信说我是positive feedback.
求offer了！听说EA的campus非常漂亮。到时候过来补onsite面经哈。



面试官中国姐姐，人真的很不错，但是一上来就让我写sql，大三学的database已经完全忘光（大概就是读file然后去掉一些不需要的
，第二题binary search tree实现查找，插入，删除。开始之前还问了不少简历上的问题，问的比较细。
早就知道他家基本不怎么招人，就当打一次酱油了，不过也算是dream company。。就当攒经验了，谁让北美老任不招国际任豚

leetcode Course ScheduleII原题
实现String的indexOf(sub)和从一个string当中去掉某一个字符remove(ch)
第一轮比较简单Java数据结构，还有hashtable什么的


电面一共45分钟。
第一题问了map reduce相关概念，具体情境下的分析。. 鐗涗汉浜戦泦,涓€浜╀笁鍒嗗湴
第二题问了BST lookup (int val)的实现
第三题BST给定lower bound和upper bound，找到BST里面在这个范围内的所有的node val
最后简单介绍了一下data platform这个组的工作内容，面试官是国人大哥，求过给offline test!
. 鍥磋鎴戜滑@1point 3 acres


补充内容 (2017-1-21 09:15):. 鐗涗汉浜戦泦,涓€浜╀笁鍒嗗湴
国人大哥好给力，过了一个小时hr followup 给了offline test:) 谢谢大哥～

补充内容 (2017-1-22 11:03):
mapreduce是问了一下什么该怎么分shard, 什么时候应该用master node, 什么时候不应该用

在全程高能的情况下自己实现了Heap, QuickSelect然后还讨论了数据量大无法一次载入内存的情况下的处理方法，也就是split and merge。
但一般这种问题都是说说就好，结果对方还是让我写了代码。写完后对方很满意，很快收到了onsite邀请。
         */
    }
}
