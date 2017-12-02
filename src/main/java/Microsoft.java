import com.sun.source.tree.Tree;
import commons.Interval;
import commons.ListNode;
import commons.TreeLinkNode;
import commons.TreeNode;

import java.util.*;

/**
 * Created by tao on 10/14/17.
 */


public class Microsoft {


    //面经题
    //replace \n to \t\n
    public String replaceN(String str){
        //return str.replace("\n","\n\t");
        int n = str.length();
        StringBuilder sb = new StringBuilder();
        for(int i=0;i<n;++i){
            if(str.charAt(i)=='\n')
                sb.append("\n\t");
            else
                sb.append(str.charAt(i));
        }
        return sb.toString();
    }

    //remove space
    public String  removeSpace(String s){
        s =  s.trim();
        StringBuilder sb = new StringBuilder();
        int n = s.length();
        for(int i=0;i<n;++i){
            if(s.charAt(i)!=' ')
                sb.append(s.charAt(i));
            else if(s.length()!=0 && sb.charAt(sb.length()-1)!=' ')
                sb.append(' ');
        }
        return sb.toString();
    }

    public void swap(int[]nums,int i,int j){
        int var = nums[i];
        nums[i]=nums[j];
        nums[j]=var;
    }
    public void wiggleSort(int[] nums) {
        int n = nums.length;
        for(int i=1;i<n;++i){
            if(i%2!=0 && nums[i]<nums[i-1])
                swap(nums,i,i-1);
            else if(i%2==0 && nums[i]>nums[i-1])
                swap(nums,i,i-1);
        }

        //也是可以现排序，然后从2开始，nums[i-1] 交换 nums[i],setp size 是2
        //也可以找到中间节点
    }

    //时针：一分钟：360／12／60=0.5 一小时：30度
    //分针： 一分钟6度
    //7:55: 时针：7*30+55*0.5=237.5
    //55*6=330, the diff  is 92.5


    public int getHeight(TreeNode root){
        if(root==null)
            return 0;
        return 1+Math.max(getHeight(root.left),getHeight(root.right));
    }

    //lowest common ancestor
    //with parent, find the path from this node to root;compare the two path
    //without parent
    public TreeNode lowestCommon(TreeNode root,TreeNode A, TreeNode B){
        if(root==null||A==root||root==B)
            return root;
        TreeNode l = lowestCommon(root.left,A,B);
        TreeNode r = lowestCommon(root.right,A,B);
        if(l!=null && r!=null)
            return root;
        else
            return l!=null?l:r;
    }

    //bst
    public TreeNode lowestCommonBST(TreeNode root,TreeNode A,TreeNode B){
        if(root==null||A==null||B==null)
            return null;
        TreeNode node =root;
        while(node!=null){
            if(node.val>Math.max(A.val,B.val))
                node = node.left;
            else if(node.val<Math.min(A.val,B.val))
                node = node.right;
            else
                break;
        }
        return node;
    }


    //palindrome integer
    public boolean isPalindromeNum(int val){
        int sum = 0;
        if(val<0||val>0 && val%10==0)
            return false;
        //if sum==0 and val==0 then it will dead loop
        while(sum<=val){
            sum=10*sum+(val%10);
            val/=10;
            if(val==sum)
                break;
        }
        return sum/10==val||val==sum;
    }
    //merge two sorted array
    //start from end
    public void merge(int[]nums1,int m,int []nums2,int n){
    }

    //binary tree populating next right pointers
    public void connect(TreeLinkNode root){
        //bfs 方法太简单了
        TreeLinkNode node =root;
        //pay attention to null
        while(node!=null && node.left!=null){
            TreeLinkNode l = node.left;
            for(;node!=null;node=node.next){
                node.left.next=node.right;
                if(node.next!=null)
                    node.right.next=node.next.left;
            }
            node=l;
        }
    }

    public void connectII(TreeLinkNode root){
        TreeLinkNode first = new TreeLinkNode(0);
        TreeLinkNode node = root;
        first.next = node;
        while(first.next!=null){
            node = first.next;
            first.next=null;
            TreeLinkNode head = first;
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
    }

    //bfs inorder successor
    public TreeNode inorderSuccessor(TreeNode root,TreeNode p){
        if(root==null||p==null)
            return null;
        if(root.val<=p.val)
            return inorderSuccessor(root.right,p);
        else{
            TreeNode left = inorderSuccessor(root.left,p);
            return left!=null?left:root;
        }
    }
    //has parent:
    //if there is no right, then find parent again and again until parent.right!=cur;
    //if has right, then right and leftmost

    //combination sum
    //居然卡了一会
    public void dfs(List<List<Integer>>res,List<Integer>path,int target,int []candidates,int ind){
        if(target==0){
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i=ind;i<candidates.length;++i){
            if(target>=candidates[i]){
                path.add(candidates[i]);
                dfs(res,path,target-candidates[i],candidates,i);
                path.remove(path.size()-1);
            }
        }
    }
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>>res = new ArrayList<>();
        dfs(res,new ArrayList<>(),target,candidates,0);
        return res;
    }


    //约瑟夫问题 count 围坐一圈，从某人开始报数，报道n出局，接着下一个人接着报数，最后剩下一个人


    //whether two rectangle is overlap or not

    //tic tac toe game


    //16 该节点是父节点的右孩子，满足节点1的最小节点
    public TreeNode getMinRightChildren(TreeNode root){
        if(root==null||root.left==null && root.right==null)
            return null;
        TreeNode l = getMinRightChildren(root.left);
        return l==null?root.right:l;
    }

    public String change(String str){
        StringBuilder sb = new StringBuilder();
        int n = str.length();
        for(int i=0;i<n;){
            if(str.charAt(i)=='a'){
                sb.append("aaa");
                i++;
            }
            else if(i<n-1 && str.charAt(i)=='b' && str.charAt(i+1)=='b'){
                sb.append('b');
                i+=2;
            }else{
                sb.append(str.charAt(i));
                i++;
            }
        }
        return sb.toString();
    }

    public String[] deleteWords(String[]args,char c){
        int n = args.length;
        int ind=0,curInd=0;
        while(curInd<n){
            if(!args[curInd].startsWith(""+c))
                args[ind++]=args[curInd];
            curInd++;
        }
        return args;
    }


    //晚上温习一遍
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer>res = new ArrayList<>();
        if(matrix.length==0||matrix[0].length==0)
            return res;
        int m = matrix.length, n = matrix[0].length;
        int curRow=0,rows=m-1,curCol=0,cols=n-1;
        while(curRow<=rows && curCol<=cols){
            for(int i=curCol;i<=cols;++i)
                res.add(matrix[curRow][i]);
            curRow++;
            for(int i=curRow;i<=rows;++i)
                res.add(matrix[i][cols]);
            cols--;
            if(rows>=curRow)
                for(int i=cols;i>=curCol;--i)
                    res.add(matrix[rows][i]);
            rows--;
            if(curCol<=cols)
                for(int i=rows;i>=curRow;--i)
                    res.add(matrix[i][curCol]);
            curCol++;
        }
        return res;
    }

    //LinkedList    n
    //detect the cycle first
    public int maxSubArray(int[]nums){
        int maxSum = Integer.MIN_VALUE;
        int cur=0;
        int start=0,end=0,ind=0,n=nums.length;
        for(int i=0;i<n;++i){
            cur+=nums[i];
            if(cur>maxSum){
                maxSum=cur;
                end=i;
                start=ind;
            }
            if(cur<0){
                cur=0;
                ind=i+1;
            }
        }
        System.out.println(start+" "+end);
        return maxSum;
    }

    //start end                   visited


    //33 clone graph
    //stack sort
    public void stackSort(Stack<Integer> stk){
        //load to array and give it back to
    }

    public void bubbleSort(int[]nums){
        int n = nums.length;
        for(int i=0;i<n;++i){
            for(int j=i+1;j<n;++j)
                if(nums[i]>nums[j]){
                    int tmp = nums[i];
                    nums[i]=nums[j];
                    nums[j]=tmp;
                }
        }
        for(int x:nums)
            System.out.println(x);
    }

    //every three
    public ListNode changeList(ListNode node){
        if(node==null||node.next==null||node.next.next==null)
            return node;
        ListNode next = node.next.next.next;
        node.next.next.next=node;
        ListNode head= node.next.next;
        node.next.next = changeList(next);
        return head;
    }

    //iterative way
    public ListNode changeListIterative(ListNode node){
        if(node==null||node.next==null||node.next.next==null)
            return node;
        ListNode first = node;
        ListNode second =node.next;
        ListNode third = node.next.next;
        ListNode head = third;
        ListNode connect = null;
        while(first!=null && second!=null && third!=null){
            ListNode next = third.next;
            third.next=first;
            second.next=next;
            if(connect!=null)
                connect.next=third;
            connect=second;
            first = next;
            second=first!=null?first.next:null;
            third = second!=null?second.next:second;
        }
        return head;
    }

    //what is .net
    //.net: A microsoft operating system platform that incorporates application,
    // a suite of tools and services and a change in the infrastructure of the company's web strategy.
    // The .net framework supports building and running of next generation of applications
    // and XML web services


    //build tree from post order
    //还有O(N)的解法
    public TreeNode build(int[]nums,int begin,int end){
        if(begin>end)
            return null;
        TreeNode root = new TreeNode(nums[end]);
        if(begin==end)
            return root;
        int ind=end-1;
        while(ind>begin){
            if(nums[ind]<nums[end])
                break;
            ind--;
        }
        root.right=build(nums,ind+1,end-1);
        root.left=build(nums,begin,ind);
        return root;
    }


    //design LRU




    //quick sort and quick select
    public void quickSort(int[]nums,int begin,int end){
        if(begin<end){
            int low =begin,hi=end,key=nums[low];
            while(low<hi){
                while(low<hi && nums[hi]>=key)
                    hi--;
                nums[low]=nums[hi];
                while(low<hi && nums[low]<=key)
                    low++;
                nums[hi]=nums[low];
            }
            nums[low]=key;
            quickSort(nums,begin,low-1);
            quickSort(nums,low+1,end);
        }
    }
    public void quickSort(int[]nums){
        quickSort(nums,0,nums.length-1);
    }


    public int quickSelect(int[]nums,int begin,int end){
            int low = begin, hi = end, key = nums[low];
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

    public int findKth(int[]nums,int begin,int end,int k){
        int ind = quickSelect(nums,begin,end);
        if(ind==k-1)
            return nums[ind];
        else if(ind>k-1)
            return findKth(nums,begin,ind,k);
        else
            return findKth(nums,ind+1,end,k);
    }
    public int findKth(int[]nums,int k){
        return findKth(nums,0,nums.length-1,k);
    }


    public void merge(int[]nums,int begin,int mid,int end){
        int[]copy = nums.clone();
        int begin1=begin,begin2=mid+1,ind=begin;
        while(begin1<=mid && begin2<=end){
            if(copy[begin1]<copy[begin2])
                nums[ind++]=copy[begin1++];
            else
                nums[ind++]=copy[begin2++];
        }
        while(begin1<=mid)
            nums[ind++]=copy[begin1++];
        while(begin2<=end)
            nums[ind++]=copy[begin2++];
    }
    public void mergeSort(int[]nums,int begin,int end){
        if(begin>=end)//这里崩了，雪崩唉
            return;
        int mid = (end-begin)/2+begin;
        mergeSort(nums,begin,mid);
        mergeSort(nums,mid+1,end);
        merge(nums,begin,mid,end);
    }
    public void mergeSort(int[]nums){
        mergeSort(nums,0,nums.length-1);
    }

    public ListNode merge(ListNode first,ListNode second){
        ListNode dummy = new ListNode(0);
        ListNode p = dummy;
        while(first!=null && second!=null){
            if(first.val<second.val){
                p.next=first;
                first = first.next;
            }else{
                p.next = second;
                second = second.next;
            }
            p=p.next;
        }
        p.next=first!=null?first:second;
        return dummy.next;
    }
    public ListNode sortList(ListNode head) {
        //merge sort;
        if(head==null||head.next==null)
            return head;
        ListNode dummy = new ListNode(0);
        dummy.next= head;
        ListNode slow = dummy;
        ListNode fast = head;
        while(fast!=null && fast.next!=null){
            fast=fast.next.next;
            slow = slow.next;
        }
        //slow.next is the last part
        ListNode second = slow.next;
        slow.next=null;
        return merge(sortList(head),sortList(second));
    }


    //45 meeting rooms
    public boolean canAttend(List<Interval>meetings){
        //sort
        Collections.sort(meetings, new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                if(o1.start!=o2.start)
                    return o1.start-o2.start;
                else
                    return o1.end-o2.end;
            }
        });
        int n = meetings.size();
        for(int i=1;i<n;++i){
            if(meetings.get(i).start<meetings.get(i-1).end)
                return false;
        }
        return true;
    }

    //priority queue
    public int minMeetingRooms(Interval[] intervals) {
        int n = intervals.length;
        int[]starts = new int[n];
        int[]ends = new int[n];
        for(int i=0;i<n;++i){
            starts[i]=intervals[i].start;
            ends[i] = intervals[i].end;
        }
        Arrays.sort(starts);
        Arrays.sort(ends);
        int cnt=0,j=0;
        for(int i=0;i<n;++i){
            if(starts[i]<ends[j]){
                cnt++;
            }else
                j++;
        }
        return cnt;
    }

    //46 walls and gates
    public void wallsAndGates(int[][] rooms) {
        //bfs
        if(rooms.length==0||rooms[0].length==0)
            return;
        int m = rooms.length, n = rooms[0].length;
        Queue<int[]>q = new LinkedList<>();
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j)
                if(rooms[i][j]==0)
                    q.offer(new int[]{i,j});
        }
        int level=1;
        int[]dx={1,-1,0,0};
        int[]dy ={0,0,1,-1};
        while(!q.isEmpty()){
            int size = q.size();
            while(size -- >0){
                int []top = q.poll();
                for(int k=0;k<4;++k){
                    int nx = top[0]+dx[k];
                    int ny = top[1]+dy[k];
                    if(nx>=0 && nx<m && ny>=0 && ny<n && rooms[nx][ny]==Integer.MAX_VALUE){
                        rooms[nx][ny]=level;
                        q.offer(new int[]{nx,ny});
                    }
                }
            }
            level++;
        }
    }


    //skyline  question


    //largest bst
    //if(!l.isBst||!r.isBst||root.val>=r.minVal||root.val<=l.maxVal){
    //res.isBst=false;

    public int evalRPN(String[] tokens) {
        int n = tokens.length;
        Stack<Integer>stk = new Stack<>();
        int a=0,b=0;
        for(int i=0;i<n;++i){
            if(tokens[i].equals("+")||tokens[i].equals("-")||tokens[i].equals("*")||tokens[i].equals("/")){
                a=stk.pop();
                b=stk.pop();
                if(tokens[i].equals("+")){
                    stk.push(a+b);
                }else if(tokens[i].equals("-")){

                    stk.push(b-a);
                }else if(tokens[i].equals("*")){

                    stk.push(a*b);
                }else if(tokens[i].equals("/")){

                    stk.push(b/a);
                }
            }
            else
                stk.push(Integer.parseInt(tokens[i]));

        }
        return stk.isEmpty()?0:stk.peek();
    }

    public void frequencySort(String s) {
        int n = s.length();
        List<Character>[] res = new ArrayList[n + 1];
        int[] cnt = new int[128];
        char[] ss = s.toCharArray();
        for (int i = 0; i < n; ++i) {
            res[i + 1] = new ArrayList<>();
            cnt[ss[i]]++;
        }
        for (int i = 0; i < 128; ++i) {
            if (cnt[i] != 0) {
                res[cnt[i]].add((char) i);
            }
        }

        for (int i = n; i > 0; --i) {
            if (!res[i].isEmpty()) {
                for (char c : res[i])
                    System.out.println("input has " + i + " " + c);
            }
        }
    }

    // a file, binary search name.

    public String getBinary(int n){
        StringBuilder sb = new StringBuilder();
        while(n!=0){
            sb.append(Math.abs(n%2));
            n>>>=1;
        }
        sb.reverse();
        //System.out.println(sb.toString());
        return sb.toString();
    }


    //build tree from string with parenthesis
    class Node{
        public int val;
        public List<Node>child;
        public boolean isNode;
        public Node(int val){
            this.val = val;
            child = new ArrayList<>();
            isNode = false;
        }
    }


    public Node createNode(String input){
        int n = input.length();
        Stack<Character>operators = new Stack<>();
        Stack<Node>nodes = new Stack<>();
        char []ss = input.toCharArray();
        for(char c:ss){
            if(c=='('){
                operators.add('(');
            }else if(c==')'){
                Node tmp = nodes.peek();
                List<Node>child = new ArrayList<>();
                while(tmp.isNode){
                    child.add(tmp);
                    nodes.pop();
                    tmp = nodes.peek();
                }
                Node newNode = new Node(tmp.val);
                nodes.pop();
                newNode.child = child;
                newNode.isNode = true;
                nodes.push(newNode);
                operators.pop();
            }else
                nodes.add(new Node(c-'0'));
        }
        return nodes.peek();
    }



    public void printMatrix(int[][]matrix){
        if(matrix==null||matrix.length==0||matrix[0].length==0)
            return;
        int m = matrix.length,n = matrix[0].length;
        int row=0, col =  n-1;
        int []dx={0,1,1,-1};
        int []dy = {-1,-1,1,1};
        System.out.println(matrix[row][col]);
        while(row>=0 && row<m && col>=0 && col<n){
            for(int k=0;k<4;++k){
                 row = row+dx[k];
                 col = col+dy[k];
                 if(row<0||row>=m||col<0||col>=n)
                     break;
                System.out.println(matrix[row][col]);
            }
        }
    }




    //behavioral questions

    //weakness:
    /*
    sometimes, I don't have a very good attention to detail, while
    that's good because it let me execute quickly, it also means that
    sometimes make careless mistakes, because of that, I make sure
    to always to have someone else double check my work.




    //execute the
     */

    /*

    what is MapReduce:
    MapReduce is a programming model and an associated implementation for processing
    big data sets with a parallel, distributed algorithms on a cluster


    word count example





    situation: Sure, I did deal with a difficult situation last semester
    when I was working on team project for my computer architecture class.
    It was a team of three people working on a improved cache replacement
    algorithm and I was in a leadership role. Things started out
    great, I organized the project, assigned tasks to team members
    However, after a few weeks, it became apparent that
    one team member in particular wasn't contributing his share

    approach: after considering how to approach the situation, I decided to have a meeting
    with him, I think it would be better to have a private conversation rather than
    confront him in front of the team.

    I scheduled a meeting between just the two of us and let him
    know how the team was reacting. During our meeting it was obvious
    he left really badly about letting the team down and he confessed
    to me that he was dedicated to the project, but was just
    overloaded at this point in the semester with exams, homework

    after thinking for a while I asked him if we could give him some time to catch up on other things,
    would he be able to catch up on our project?  He assured me he could. so I rearrange the task.
    as a result he did eventually perform well for the team and made a strong contribution to our project."






    1. given an app, if the app crash for some reason, how would you debug it to locate the problems.

    the first thing to do is : do not panic, you are likely to worsen the situation if you freak out
    and start changing things at random.

    the second thing is try to find the bug in which file and on which line.
    you can customize some logs and outputs, such as, print "enter this function" in the beginning of
    this function and print"leave this function" in the end of this function. And then you can analysis
    the logs and console output to find the what's the bug and where's the bug.

    do some test for the part of code that has bugs.

    2. 如何处理坑队友的问题，你要直接去骂他吗还是？ =》 我答的是，如果人很nice，open minded的直接跟他说， 如果人很difficult，跟他身边的朋友说，让他的朋友提醒他



    3. 如果给你很难的problem， 你肯定是要撞墙的？ 你会怎么样应对？

    make a web application:
    analysis the requirement first: know how many requests the website should deal with, if it is small, you can use python flask, if it is large,
    you can use java spring. know query per second, so you can decide the way to design database, has index or not.
    during the process, I encounter many problems, so I will ask my mentor for help or use internet to find solutions.

    Once you have done requirement analysis, you can start to program.




    4. why ms
    I surf the internet on the windows since I was a child; I create documents, excel, slides by office;
    I write c++ code on the visual studio; I played games on the Xbox one; I cannot imagine my life
    without microsoft. To me, microsoft is one of few greatest companies who change the world.

    5. 怎么分工，遇到矛盾如何解决




    6. 最难的一次project和组内如何分工，有人不做事怎么办，自己propose被否决怎么办

    " I will schedule a private meeting with him and express my opinion logically, reasonably and professionally
     and will try to resolve our dispute in a peaceful manner.
     We both agreed that our goal was to make the team better and came to a compromise that consisted of both of our ideas."


    7. 说自己做的project， 团队做pj的经历，如何分工，有无leader。谁是leader


    8： behavioral conflict


    9: 遇到死脑筋的人


    10： data migration


    11. why this role
    this role is perfectly aligned with where I want to grow my career and what I have been working towards,
    and given my skills and experiences, I feel i can make significant contributions to both this team
    and company, and as part of the overall team, help drive the company to the next level.
     */



}




