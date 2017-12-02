
import commons.*;

import java.util.*;



class UnionFind{
    private int[]parent = null;
    private int[]rank = null;
    public UnionFind(int n){
        parent = new int[n+1];
        rank = new int[n+1];
        for(int i=0;i<=n;++i){
            parent[i]=i;
        }
    }

    public int find(int x){
        while(x!=parent[x]){
            parent[x]=parent[parent[x]];
            x=parent[x];
        }
        return x;
    }

    public boolean mix(int x,int y){
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
}

//LRU
//delete the key in the map
// map.get(key);
class LRUCache {
    DoubleListNode head = null;
    DoubleListNode tail = null;
    Map<Integer,DoubleListNode>map = null;
    private int _capacity;
    public LRUCache(int capacity) {
        head = new DoubleListNode(-1,-1);
        tail = new DoubleListNode(-1,-1);
        map = new HashMap<>();
        head.next = tail;
        tail.prev = head;
        _capacity = capacity;
    }

    public int get(int key) {
        if(!map.containsKey(key))
            return -1;
        int val = map.get(key).val;
        unlink(map.get(key));
        insertInHead(map.get(key));
        return val;
    }

    public void put(int key, int value) {
        if(map.containsKey(key)){
            DoubleListNode node = map.get(key);
            node.val = value;
            unlink(node);
            insertInHead(map.get(key));
        }else{
            if(_capacity<=map.size()){
                DoubleListNode deleteNode = tail.prev;
                unlink(deleteNode);
                map.remove(deleteNode.key);
            }
            map.put(key,new DoubleListNode(key,value));
            insertInHead(map.get(key));
        }
    }

    public void unlink(DoubleListNode node){
        node.next.prev = node.prev;
        node.prev.next = node.next;
        node.next=null;
        node.prev = null;
    }

    public void insertInHead(DoubleListNode node){
        node.next = head.next;
        head.next.prev = node;
        node.prev = head;
        head.next = node;
    }
}


class MedianFinder {

    /** initialize your data structure here. */
    PriorityQueue<Integer>pq1=null;
    PriorityQueue<Integer>pq2=null;
    public MedianFinder() {
        pq1=new PriorityQueue<>(Collections.reverseOrder());//smaller part
        pq2=new PriorityQueue<>();//larger part
    }

    public void addNum(int num) {
        if(pq1.isEmpty()||pq1.peek()<=num)
            pq1.add(num);
        else
            pq2.add(num);
        while(pq1.size()>pq2.size()+1){
            pq2.offer(pq1.poll());
        }
        while(pq2.size()>pq1.size()){
            pq1.offer(pq2.poll());
        }
    }

    public double findMedian() {
        int n = pq1.size()+pq2.size();
        return n%2==0?(pq1.peek()+pq2.peek())/2.0:pq1.peek()*1.0;
    }
}
//就是一个inorder 的iterator
class BSTIterator {
    private Stack<TreeNode>stk = null;
    private TreeNode cur = null;
    public BSTIterator(TreeNode root) {
        cur = root;
        stk = new Stack<>();
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return cur!=null||!stk.isEmpty();
    }

    /** @return the next smallest number */
    public int next() {
        while(cur!=null){
            stk.push(cur);
            cur = cur.left;
        }
        cur = stk.pop();
        int val = cur.val;
        cur = cur.right;
        return val;
    }
}
class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        serialize(root,sb);
        return sb.toString();
    }

    private void serialize(TreeNode root,StringBuilder sb){
        if(root==null){
            sb.append('@').append(' ');
            return;
        }
        sb.append(root.val).append(' ');
        serialize(root.left,sb);
        serialize(root.right,sb);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String []strs = data.split(" ");
        Deque<String>dq =new LinkedList<>();
        dq.addAll(Arrays.asList(strs));
        return deserialize(dq);
    }

    private TreeNode deserialize(Deque<String>dq){
        if(dq.isEmpty())
            return null;
        String top = dq.pollFirst();
        if(top.equals("@"))
            return null;
        TreeNode root = new TreeNode(Integer.parseInt(top));
        root.left = deserialize(dq);
        root.right = deserialize(dq);
        return root;
    }
}

public class MicrosoftOnsite {
    //tree questions
    //101 Symmetric Tree

    //main idea, compare the left.left, right.right and left.right, right.left
    public boolean isSymmetric(TreeNode left, TreeNode right){
        if(left==null||right==null)
            return left==right;
        return left.val==right.val && isSymmetric(left.left,right.right) && isSymmetric(left.right,right.left);
    }
    public boolean isSymmetric(TreeNode root) {
        if(root==null)
            return true;
        return isSymmetric(root.left,root.right);
    }

    //queue two queue
    public boolean isSymmetricByQueue(TreeNode root){
        if(root==null)
            return true;
        Queue<TreeNode> p = new LinkedList<>();
        Queue<TreeNode> q = new LinkedList<>();
        if(root.left!=null)
            p.offer(root.left);
        if(root.right!=null)
            q.offer(root.right);
        if(p.size()!=q.size())
            return false;
        while(!q.isEmpty() && !p.isEmpty()){
            TreeNode pNode = p.poll();
            TreeNode qNode = q.poll();
            if(pNode.val!=qNode.val)
                return false;
            if(pNode.left!=null)
                p.offer(pNode.left);
            if(qNode.right!=null)
                q.offer(qNode.right);
            if(p.size()!=q.size())
                return false;
            if(pNode.right!=null)
                p.offer(pNode.right);
            if(qNode.left!=null)
                q.offer(qNode.left);
            if(p.size()!=q.size())
                return false;
        }
        return true;
    }

    public boolean hasPathSum(TreeNode root, int sum) {
        if(root==null)
            return false;
        if(root.val==sum && root.left==null && root.right==null)
            return true;
        return hasPathSum(root.left,sum-root.val)||hasPathSum(root.right,sum-root.val);
    }
    //iterative way
    class Tuple{
        public TreeNode node;
        public int val;
        public Tuple(TreeNode node,int val){
            this.val = val;
            this.node = node;
        }
    }
    public boolean hasPathSumIterative(TreeNode root,int sum){
        if(root==null)
            return false;
        Queue<Tuple>q = new LinkedList<>();
        q.offer(new Tuple(root,root.val));
        while(!q.isEmpty()){
            Tuple top = q.poll();
            if(top.node.left==null && top.node.right==null && top.val==sum)
                return true;
            if(top.node.left!=null){
                q.offer(new Tuple(top.node.left,top.val+top.node.left.val));
            }
            if(top.node.right!=null){
                q.offer(new Tuple(top.node.right,top.val+top.node.right.val));
            }
        }
        return false;
    }



    //236 Lowest Common Ancestor of a Binary Tree
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root==null||root==p||root==q)
            return root;
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        if(left!=null && right!=null)
            return root;
        return left!=null?left:right;

    }

    //dfs find two path and last common;
    public boolean dfs(TreeNode root,TreeNode p,List<TreeNode>ans){
        if(root==null)
            return false;
        if(root==p){
            ans.add(root);
            return true;
        }
        ans.add(root);
        if(dfs(root.left,p,ans))
            return true;
        if(dfs(root.right,p,ans))
            return true;
        ans.remove(ans.size()-1);
        return false;
    }

    public TreeNode lowestCommonAncestorAnotherWay(TreeNode root, TreeNode p, TreeNode q) {

        TreeNode node = root;
        List<TreeNode>pList = new ArrayList<>();
        List<TreeNode>qList = new ArrayList<>();
        dfs(node,p,pList);
        node = root;
        dfs(node,q,qList);
        int ind =0, m = qList.size(),n = pList.size();
        while(ind<m && ind<n){
            if(pList.get(ind)!=qList.get(ind))
                break;
            ind++;
        }
        if(ind-1<0)
            return null;
        return qList.get(ind-1);
    }


    //114
    TreeNode pre = null;
    public void flatten(TreeNode root) {
        if(root==null)
            return;
        flatten(root.right);
        flatten(root.left);
        root.right = pre;
        root.left = null;
        pre = root;
    }

    //iterative
    public void flattenIterative(TreeNode root) {
        if(root==null)
            return;
        TreeNode node = root;
        while(node!=null){
            if(node.left!=null){
                //right most;
                TreeNode rightMost = node.left;
                while(rightMost.right!=null)
                    rightMost=rightMost.right;
                rightMost.right = node.right;
                node.right = node.left;
                node.left = null;
            }
            node = node.right;
        }
    }


    public TreeNode buildTree(int[]inorder,int start1,int end1,int[]postorder,int start2,int end2,Map<Integer,Integer>map){
        if(start2>end2)
            return null;
        if(start2==end2){
            return new TreeNode(postorder[end2]);
        }
        TreeNode root = new TreeNode(postorder[end2]);
        //find the root in the inorder;
        int ind = map.get(postorder[end2]);
        root.left = buildTree(inorder,start1,ind-1,postorder,start2,ind-1-start1+start2,map);
        root.right = buildTree(inorder,ind+1,end1,postorder,ind-start1+start2,end2-1,map);
        return root;
    }
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        int n = postorder.length;
        //打表处理，用map store first
        Map<Integer,Integer>map =new HashMap<>();
        return buildTree(inorder,0,n-1,postorder,0,n-1,map);
    }
    //save space




    //103 Binary Tree Zigzag Level Order Traversal
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>>ans = new ArrayList<>();
        Queue<TreeNode>q = new LinkedList<>();
        if(root!=null)
            q.offer(root);
        while(q.isEmpty()){
            int size = q.size();
            List<Integer>level = new ArrayList<>();
            while(size -->0){
                TreeNode top = q.poll();
                level.add(top.val);
                if(top.left!=null)
                    q.offer(top.left);
                if(top.right!=null)
                    q.offer(top.right);
            }
            if(ans.size()%2!=0)
                Collections.reverse(level);
            ans.add(level);
        }
        return ans;
    }

    //reverse the output
    public static void dfs(List<List<Integer>>ans,int level,TreeNode root){
        if(root==null)
            return;
        if(level>=ans.size())
            ans.add(0,new ArrayList<>());
        ans.get(ans.size()-level-1).add(root.val);
        dfs(ans,level+1,root.left);
        dfs(ans,level+1,root.right);
    }
    public static void reverseLevelOrder(TreeNode root){
        //dfs
        List<List<Integer>>ans = new ArrayList<>();
        dfs(ans,0,root);
        System.out.println(ans);
    }

    //dfs way
    public void dfs(TreeNode root,List<List<Integer>>ans,int ind){
        if(root==null)
            return;
        if(ans.size()<=ind)
            ans.add(new ArrayList<>());
        if(ans.size()%2==0)
            ans.get(ind).add(0,root.val);
        else
            ans.get(ind).add(root.val);
        dfs(root.left,ans,ind+1);
        dfs(root.right,ans,ind+1);
    }
    public List<List<Integer>>zigzagLevelOrderRecursive(TreeNode root){
        List<List<Integer>>ans = new ArrayList<>();
        dfs(root,ans,0);
        return ans;
    }


    //124
    public int dfs(TreeNode root,int[]ans){
        if(root==null)
            return 0;
        int l = dfs(root.left,ans);
        int r =dfs(root.right,ans);
        int res = Math.max(root.val,Math.max(root.val+l,root.val+r));
        ans[0]=Math.max(ans[0],Math.max(res,root.val+l+r));
        return res;
    }
    public int maxPathSum(TreeNode root) {
        int []ans = {-2147483648};
        dfs(root,ans);
        return ans[0];
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>>ans =new ArrayList<>();
        Queue<TreeNode>q = new LinkedList<>();
        if(root!=null)
            q.offer(root);
        while(!q.isEmpty()){
            int size = q.size();
            List<Integer>level = new ArrayList<>();
            while(size -->0){
                TreeNode top = q.poll();
                level.add(top.val);
                if(top.left!=null)
                    q.offer(top.left);
                if(top.right!=null)
                    q.offer(top.right);
            }
            ans.add(level);
        }
        return ans;
    }

    public void dfs102(TreeNode root,int ind,List<List<Integer>>ans){
        if(root==null)
            return;
        if(ans.size()<=ind)
            ans.add(new ArrayList<>());
        ans.get(ind).add(root.val);
        dfs102(root.left,ind+1,ans);
        dfs102(root.right,ind+1,ans);
    }
    public List<List<Integer>>levelOrderRecursive(TreeNode root){
        List<List<Integer>>ans =new ArrayList<>();
        dfs102(root,0,ans);
        return ans;
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer>ans = new ArrayList<>();
        Stack<TreeNode>stk =new Stack<>();
        TreeNode cur = root;
        while(cur!=null||!stk.isEmpty()){
            while(cur!=null){
                stk.push(cur);
                cur = cur.left;
            }
            cur = stk.pop();
            ans.add(cur.val);
            cur=cur.right;
        }
        return ans;
    }


    //98. Validate Binary Search Tree
    //跑一个stk的pre
    //表示范围的搞起来
    public boolean isValidBST(TreeNode root,Integer low,Integer hi){
        if(root==null)
            return true;
        return (low==null||low<root.val) && (hi==null||hi>root.val) && isValidBST(root.left,low,root.val) && isValidBST(root.right,root.val,hi);
    }
    public boolean isValidBST(TreeNode root) {
        return isValidBST(root,null,null);
    }


    public TreeNode inorderSuccessor(TreeNode root,TreeNode p){
        if(root==null)
            return null;
        if(root.val<=p.val)
            return inorderSuccessor(root.right,p);
        else{
            TreeNode l = inorderSuccessor(root.left,p);
            return l==null?root:l;
        }
    }
    public TreeNode inorderSuccessorIteratvie(TreeNode root,TreeNode p){
        if(root==null)
            return null;
        TreeNode successor = null;
        while(root!=null){
            if(root.val<=p.val)
                root=root.right;
            else{
                successor = root;
                root=root.left;
            }
        }
        return successor;
    }


    //bfs is not necessary
    public void connect(TreeLinkNode root) {
        TreeLinkNode first = new TreeLinkNode(0);
        first.next = root;
        while(first.next!=null){
            TreeLinkNode head = first.next;
            TreeLinkNode node = first;
            first.next = null;
            for(;head!=null;head=head.next){
                if(head.left!=null){
                    node.next = head.left;
                    node = node.next;
                }
                if(head.right!=null){
                    node.next = head.right;
                    node = node.next;
                }
            }
        }
        System.out.println("finish");
    }


    public void dfs(List<List<Integer>>ans,int[]nums,List<Integer>path){
        if(path.size()==nums.length){
            ans.add(new ArrayList<>(path));
            return;
        }
        for(int i=0;i<nums.length;++i){
            if(!path.contains(nums[i])){
                path.add(nums[i]);
                dfs(ans,nums,path);
                path.remove(path.size()-1);
            }
        }
    }
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>>ans = new ArrayList<>();
        dfs(ans,nums,new ArrayList<>());
        return ans;
    }

    //swap way

    public void swap(int[]nums,int x,int y){
        int tmp = nums[x];
        nums[x] = nums[y];
        nums[y]=tmp;
    }
    public void dfs(List<List<Integer>>ans,int ind,int[]nums){
        if(ind==nums.length){
            List<Integer>one = new ArrayList<>();
            for(int x:nums)
                one.add(x);
            ans.add(one);
            return;
        }
        for(int i=ind;i<nums.length;++i){
            swap(nums,i,ind);
            dfs(ans,i+1,nums);
            swap(nums,i,ind);
        }
    }
    public List<List<Integer>>permuteSwap(int[]nums){
        List<List<Integer>>ans = new ArrayList<>();
        dfs(ans,0,nums);
        return ans;
    }


    public void dfs47(List<List<Integer>>ans,int ind,int[]nums){
        if(ind==nums.length){
            List<Integer>one = new ArrayList<>();
            for(int x:nums)
                one.add(x);
            ans.add(one);
            return;
        }
        Set<Integer>appeared = new HashSet<>();
        for(int i=ind;i<nums.length;++i){
            if(appeared.add(nums[i])){
            System.out.println("i: "+i+" ind: "+ind);
            swap(nums,i,ind);
            dfs47(ans,ind+1,nums);
            swap(nums,i,ind);
            }
        }
    }
    public List<List<Integer>>permuteUnique(int[]nums){
        List<List<Integer>>ans = new ArrayList<>();
        Arrays.sort(nums);
        dfs47(ans,0,nums);
        return ans;
    }

    //vis array 在外面，而且可以检测到true直接返回，并不需要vis[nx][ny]=false;
    //可以好好想想的
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

    //48 rotate image
    //brute force
    public void rotate(int[][] matrix) {
        if(matrix==null||matrix.length==0||matrix[0].length==0)
            return;
        int m = matrix.length,n = matrix[0].length;
        int [][]copy = new int[m][n];
        for(int i=0;i<m;++i)
            copy[i]=matrix[i].clone();
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                matrix[i][j]=copy[j][n-1-i];
            }
        }
    }

    //anti-close
    public static void reverse(int[]nums){
        int n = nums.length;
        int begin=0, end=n-1;
        while(begin<end){
            int tmp = nums[begin];
            nums[begin++]=nums[end];
            nums[end--]=tmp;
        }
    }
    public static void rotateAnti(){
        Scanner scanner = new Scanner(System.in);
        int T = scanner.nextInt();
        while(T-- >0){
            int n = scanner.nextInt();
            int [][]matrix = new int[n][n];
            int m = n;
            n*=n;
            for(int i=0;i<n;++i){
                matrix[i/m][i%m]=scanner.nextInt();
            }
            for(int i=0;i<m;++i){
                reverse(matrix[i]);
            }
            for(int i=0;i<m;++i){
                for(int j=0;j<i;++j){
                    int tmp = matrix[i][j];
                    matrix[i][j]=matrix[j][i];
                    matrix[j][i]=tmp;
                }
            }
            for(int i=0;i<m;++i){
                for(int j=0;j<m;++j)
                    System.out.print(matrix[i][j]+ " ");
            }
            System.out.println();
        }
    }

    public void rotateSaveSpace(int[][]matrix){
        if(matrix==null||matrix.length==0||matrix[0].length==0)
            return;
        int n = matrix.length;
        int i=0,j=n-1;
        while(i<j){
            int []tmp = matrix[i].clone();
            matrix[i]=matrix[j].clone();
            matrix[j]=tmp.clone();
        }
        for(i=0;i<n;++i){
            for(j=0;j<i;++j){
                int num = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i]=num;
            }
        }
    }


    //why reverse order;
    /*
    1 0 2 3 4
    4 5 6 7 8

    the zero in the first line would lead to zeroes everywhere
     */
    public void setZeroes(int[][] matrix) {
        if(matrix==null||matrix.length==0||matrix[0].length==0)
            return;
        int m = matrix.length, n = matrix[0].length;
        boolean colHasZero = false;
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(j==0){
                    if(matrix[i][j]==0)
                        colHasZero = true;
                    continue;
                }
                if(matrix[i][j]==0){
                    matrix[i][0]=0;
                    matrix[0][j]=0;
                }
            }
        }

        for(int i=m-1;i>=0;--i){
            for(int j=n-1;j>=1;--j){
                if(matrix[i][0]==0 || matrix[0][j]==0)
                    matrix[i][j]=0;
            }
            if(colHasZero)
                matrix[i][0]=0;
        }
    }



    //row and rows are the low bound and upper bound
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer>ans = new ArrayList<>();
        if(matrix==null||matrix.length==0||matrix[0].length==0)
            return ans;
        int rows = matrix.length-1, cols = matrix[0].length-1;
        int row=0,col=0;
        while(row<=rows && col<=cols){
            for(int i=col;i<=cols;++i){
                ans.add(matrix[row][i]);
            }
            row++;
            for(int i=row;i<=rows;++i){
                ans.add(matrix[i][cols]);
            }
            cols--;

            if(row<=rows){
                for(int i=cols;i>=col;--i){
                    ans.add(matrix[rows][i]);
                }
                rows--;
            }
            if(col<=cols){
                for(int i=rows;i>=row;--i){
                    ans.add(matrix[i][col]);
                }
                col++;
            }
        }
        return ans;
    }

    //24 swap nodes in pairs
    //recursive ways
    public ListNode swapPairs(ListNode head) {
        if(head==null||head.next==null)
            return head;
        ListNode remaining = swapPairs(head.next.next);
        ListNode ans = head.next;
        head.next.next = head;
        head.next = remaining;
        return ans;
    }

    //iterative ways
    public ListNode swapPairsIterative(ListNode head) {
        if(head==null||head.next==null)
            return head;
        ListNode ans = head.next;
        ListNode first = head;
        ListNode second = head.next;
        ListNode preHead = null;
        while(first!=null && second!=null){
            first.next = second.next;
            second.next = first;
            if(preHead!=null)
                preHead.next = second;
            preHead = first;
            if(first!=null)
                first = first.next;
            if(first!=null)
                second = first.next;
        }

        return ans;
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode p = dummy;
        while(l1!=null && l2!=null){
            if(l1.val>=l2.val){
                p.next = l2;
                l2 = l2.next;
            }else{
                p.next = l1;
                l1 = l1.next;
            }
            p = p.next;
        }
        if(l1!=null)
            p.next = l1;
        if(l2!=null)
            p.next = l2;
        if(p!=null)
            p.next = null;
        return dummy.next;
    }


    //23 merge k sorted list
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode>pq = new PriorityQueue<>(new Comparator<ListNode>() {
            @Override
            public int compare(ListNode o1, ListNode o2) {
                return o1.val-o2.val;
            }
        });
        for(ListNode list:lists){
            if(list!=null)
                pq.offer(list);
        }

        ListNode dummy = new ListNode(-1);
        ListNode p = dummy;
        while(!pq.isEmpty()){
           ListNode top = pq.poll();
           p.next = top;
           p = p.next;
           if(top.next!=null)
               pq.offer(top.next);
        }
        return dummy.next;
    }

    // intersection of two linked list
    //attention to the usage; you should check the p==q before the p==null && q==null
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA==null||headB==null)
            return null;
        ListNode p = headA;
        ListNode q = headB;
        while(p!=q){
            p=p.next;
            q=q.next;
            if(p==q)
                break;
            if(p==null)
                p=headB;
            if(q==null)
                q=headA;
        }
        return p;
    }

    //25 reverse nodes in k-group
    public ListNode reverseList(ListNode node){
        if(node==null||node.next==null)
            return node;
        ListNode newHead = null;
        while(node!=null){
            ListNode next = node.next;
            node.next = newHead;
            newHead = node;
            node = next;
        }
        return newHead;
    }
    public ListNode reverseKGroup(ListNode head, int k) {
        if(k<=1||head==null||head.next==null)
            return head;
        ListNode p = head;
        for(int i=1;i<k && p!=null;++i)
            p=p.next;
        if(p==null)
            return head;
        ListNode next = reverseKGroup(p.next,k);
        p.next=null;
        ListNode ans = reverseList(head);
        head.next = next;
        return ans;
    }



    //232 implement queue using stacks
    //Amortized Analysis  push O(1),  amortized pop(1), peek o(1), isEmpty();//o(1)


    //71 simplify path
    //very important
    //pay attention to the ".." when stk is empty
    public String simplifyPath(String path) {
        String []paths = path.split("/");
        Stack<String>stk = new Stack<>();
        for(String str:paths){
            if(str.isEmpty()||str.equals("/")||str.equals("."))
                continue;
            if(!stk.isEmpty() && str.equals(".."))
                stk.pop();
            else if(!str.equals("/"))
                stk.push(str);
        }
        List<String>ans = new ArrayList<>(stk);
        return ans.isEmpty()?"/":String.join("/",ans);
    }

    //162 find the peak element
    //
    public int findPeakElement(int[] nums) {
        int begin = 0, end = nums.length-1;
        while(begin<end){
            int mid = (end-begin)/2+begin;
            if(nums[mid]<nums[mid+1]){
                begin=mid+1;
            }else
                end=mid;
        }
        return nums[begin];
    }


    //153 find the min in the rotated sorted array
    //一直和search in rotated sorted array 搞混了，，居然还会搞混，草
    public int findMin(int[] nums) {
        int begin =0 , end =nums.length-1;
        while(begin<end){
            if(nums[begin]<nums[end])
                return nums[begin];
            int mid = (end-begin)/2+begin;
            if(nums[mid]<nums[end])
                end=mid;
            else
                begin=mid+1;
        }
        return nums[begin];
    }

    //125 valid palindrome
    public boolean isPalindrome(String s) {
        char []ss=s.toCharArray();
        int begin =0 , end=ss.length-1;
        while(begin<end){
            while(begin<end && !Character.isLetterOrDigit(ss[begin]))
                begin++;
            while(begin<end && !Character.isLetterOrDigit(ss[end]))
                end--;
            if(begin<end){
                if(Character.toLowerCase(ss[begin])!=Character.toLowerCase(ss[end]))
                    return false;
                begin++;
                end--;
            }
        }
        return true;
    }

    //55 jump game
    //很有意思
    //return reach>=n-1;
    public boolean canJump(int[] nums) {
        int reach=0,n=nums.length,i=0;
        for(reach=nums[0];i<n && i+nums[i]<=reach;++i){
            reach=Math.max(i+nums[i],reach);
        }
        return i>=n;
    }

    //longest palindrome substring
    public String longestPalindrome(String s) {
        int n = s.length();
        boolean [][]dp = new boolean[n+1][n+1];
        dp[0][0]=true;
        for(int i=1;i<=n;++i){
            for(int j=i;j>=1;--j){
                if(s.charAt(i)==s.charAt(j) && (j+1>i-1||dp[j+1][i-1])){
                    dp[j][i]=true;
                }
            }
        }
        int len=0,start=0;
        for(int i=1;i<=n;++i){
            for(int j=i;j<=n;++j){
                if(dp[i][j] && len<j-i+1){
                    len=j-i+1;
                    start = i-1;
                }
            }
        }
        return s.substring(start,start+len);
    }

    //extend in two directions;
    public int extendAroundCenter(String s,int start1,int start2){
        int len=0,n=s.length();
        while(start1>=0 && start2<n){
            if(s.charAt(start1)!=s.charAt(start2))
                break;
            if(start1==start2)
                len++;
            else
                len+=2;
            start1--;
            start2++;
        }
        return len;
    }
    public String longestPalindromeSubstring(String s){
        int n = s.length();
        int len=0,start=0;
        for(int i=0;i<n;++i){
            int length = extendAroundCenter(s,i,i);
            if(len<length){
                len=length;
                start = i-len/2;
            }

            length = extendAroundCenter(s,i,i);
            if(len<length){
                len=length;
                start = i-len/2+1;
            }
        }
        return s.substring(start,len);
    }


    //length of LIS
    //O(n^2)
    public int lengthOfLIS(int[] nums) {
        int n = nums.length;
        int []dp=new int[n];
        int len=0;
        for(int i=0;i<n;++i){
            dp[i]=1;
            for(int j=i-1;j>=0;--j){
                if(nums[i]>nums[j] && dp[i]<dp[j]+1)
                    dp[i]=dp[j]+1;
            }
            len=Math.max(dp[i],len);
        }
        return len;
    }
    //O(nlogn)
    public int lowerBound(List<Integer>nums,int target){
        int n = nums.size();
        int begin=0,end=n-1;
        if(target>nums.get(end))
            return n;
        while(begin<end){
            int mid = (end-begin)/2+begin;
            if(nums.get(mid)>=target)
                end=mid;
            else
                begin=mid+1;
        }
        return begin;
    }
    public int lengthOfLISBYBinarySearch(int[]nums){
        List<Integer>ans = new ArrayList<>();
        int n = nums.length;
        for(int i=0;i<n;++i){
            if(ans.isEmpty() ||ans.get(ans.size()-1)<nums[i]){
                ans.add(nums[i]);
            }else{
                int ind = lowerBound(ans,nums[i]);
                ans.set(ind,nums[i]);
            }
        }
        return ans.size();
    }

    //house robber ii
    public int rob(int[] nums) {
        int n=nums.length;
        if(n==0)
            return 0;
        if(n==1)
            return nums[0];
        if(n==2)
            return Math.max(nums[0],nums[1]);
        int[]dp1=new int[n-1];
        int []dp2 = new int[n-1];
        dp1[0]=nums[1];
        dp2[0]=nums[0];
        //dp1 start from index 1 to n-1
        //dp2 start from index 0 to n-2;
        for(int i=2;i<=n-1;++i){
            dp1[i-1]=Math.max(dp1[i-2],(i>=3?dp1[i-3]:0)+nums[i]);
            dp2[i-1]=Math.max(dp2[i-2],(i>=3?dp2[i-3]:0)+nums[i-1]);
        }
        return Math.max(dp1[n-2],dp2[n-2]);
    }

    //house robber II
    public int[]dfs(TreeNode root){
        int[]ans ={0,0};
        if(root==null)
            return ans;
        int []l = dfs(root.left);
        int []r = dfs(root.right);
        //ans[0], does not have root, ans[1] has root;
        ans[1]=root.val+l[0]+r[0];
        ans[0]=Math.max(l[0],l[1])+Math.max(r[0],r[1]);
        return ans;
    }
    public int rob(TreeNode root) {
        int []ans = dfs(root);
        return Math.max(ans[0],ans[1]);
    }


    //decodes way II,要好好想想
    public int numDecodings(String s) {
        int n = s.length();
        if(n==0||s.charAt(0)=='0')
            return 0;
        int []dp=new int[n+1];
        //think louder
        dp[1]=1;
        dp[0]=1;
        for(int i=2;i<=n;++i){
            if(s.charAt(i-1)=='0'){
                if(s.charAt(i-2)=='0'||s.charAt(i-2)>'2')
                    return 0;
                else
                    dp[i]=dp[i-2];
            }else{
                if(s.charAt(i-2)=='1'||(s.charAt(i-2)=='2' && s.charAt(i-1)>='1' && s.charAt(i-1)<='6'))
                    dp[i]=dp[i-1]+dp[i-2];
                else
                    dp[i]=dp[i-1];
            }
        }
        return dp[n];
    }


    public void rotate(int[] nums, int k) {
        int n = nums.length;
        int []clone = nums.clone();
        for(int i=0;i<n;++i)
            nums[(i+k)%n]=clone[i];
    }

    //三步反转法
    public void reverse(int[]nums,int begin,int end){
        end--;
        while(begin<end){
            int tmp = nums[begin];
            nums[begin++]=nums[end];
            nums[end--]=tmp;
        }
    }
    public void rotateReverse(int[]nums,int k){
        int n = nums.length;
        k%=n;
        k=n-k;
        reverse(nums,0,k);
        reverse(nums,k,n);
        reverse(nums,0,n);
    }


    public int gcd(int a,int b){
        return b==0?a:gcd(b,a%b);
    }

    public void rotateByGCD(int[]nums,int k){
        int n = nums.length;
        int gcd = gcd(n,k);
        k%=n;
        for(int i=0;i<gcd;++i){
            int save = nums[i];
            int next = i;
            while(((next-k+n)%n)!=i){
                nums[next]=nums[(next-k+n)%n];
                next = (next-k+n)%n;
            }
            nums[next]=save;
        }
    }


    //258 add digits;
    public int addDigits(int num){
        //n = a0+a1*10^1 +...+an*10^n;
        //n%9 = a0%9+a1%9+...an%9;
        //1-9
        //n%9 - 0,8
        //so we add 1+(n-1)%9;
        return 1+(num-1)%9;
    }



    public int strStr(String haystack, String needle) {
        if(needle.length()>haystack.length())
            return -1;
        int m = haystack.length(), n = needle.length();
        for(int i=0;i<=m-n;++i){
            int j=0;
            for(;j<n;++j){
                if(haystack.charAt(i+j)!=needle.charAt(j))
                    break;
            }
            if(j==n)
                return i;
        }
        return -1;
    }

    //kmp
    //find the next array
    //p0 p1, ..., pk-1 = pj-k pj-k+1, ..., pj-1
    public int[] getNext(String pattern){
        int n = pattern.length();
        int []next = new int[n+1];
        next[0]=-1;
        int k=-1;
        int j=0;
        while(j<n){
            if(k==-1 || pattern.charAt(j)== pattern.charAt(k)){
                k++;
                j++;
                next[j]=k;
            }else
                k=next[k];
        }
        return next;

    }

    public int strStrKMP(String haystack,String needle){
        int m = haystack.length(), n = needle.length();
        int i=0,j=0;
        int []next = getNext(needle);
        while(i<m && j<n){
            if(j==-1||haystack.charAt(i)==needle.charAt(j)){
                i++;
                j++;
            }else{
                j=next[j];
            }
        }
        return j==n?i-j:-1;
    }

    //121 best time to buy and sell stock
    public int maxProfit(int[] prices) {
        int n = prices.length;
        if(n<=1)
            return 0;
        int minPrice=prices[0];
        int maxVal = 0;
        for(int i=1;i<n;++i){
            minPrice=Math.min(minPrice,prices[i]);
            maxVal = Math.max(maxVal,prices[i]-minPrice);
        }
        return maxVal;
    }



    //188 Best Time to Buy and Sell Stock IV
    //You may complete at most k transactions.
    public int maxProfit(int k, int[] prices) {
        return 0;
    }


    //two transactions;
    public int maxProfitTwoTransaction(int[] prices) {
        int n = prices.length;
        int []first = new int[n];
        int []second = new int[n];
        int price=Integer.MAX_VALUE;
        for(int i=0;i<n;++i){
            price = Math.min(price,prices[i]);
            first[i]=Math.max(i>=1?first[i-1]:0,prices[i]-price);//first[i] 和 first[i-1]有关系
        }
        price = Integer.MIN_VALUE;
        int val=0;
        for(int i=n-1;i>=0;--i){
            price = Math.max(price,prices[i]);
            second[i] = Math.max(i<=n-2?second[i+1]:0,price-prices[i]);
            val=Math.max(val,second[i]+first[i]);
        }
        return val;
    }
    // two transactions
    public int maxProfitTwoTransactionsConcise(int[]prices){
        int n = prices.length;
        if(n<=1)
            return 0;
        int secondBuy = -prices[0];
        int secondSell=0;
        int firstBuy=-prices[0];
        int firstSell=0;
        for(int x:prices){
            secondSell=Math.max(secondSell,secondBuy+x);
            secondBuy = Math.max(secondBuy,firstSell-x);
            firstSell = Math.max(firstSell, firstBuy+x);
            firstBuy = Math.max(firstBuy,-x);
        }
        return secondSell;
    }


    //k transactions
    private int quickSolve(int[] prices) {
        int len = prices.length, profit = 0;
        for (int i = 1; i < len; i++)
            // as long as there is a price gap, we gain a profit.
            if (prices[i] > prices[i - 1]) profit += prices[i] - prices[i - 1];
        return profit;
    }
    //受到启发，其实这道题和post office很像
    public int maxProfitKTransactions(int k, int[] prices) {
        int n = prices.length;
        if(n<=1||k<=0)
            return 0;
        if (k >= n / 2)
            return quickSolve(prices);
        int []sells = new int[k];
        int []buys = new int[k];
        Arrays.fill(buys,-prices[0]);
        for(int x:prices){
            for(int i=k-1;i>=0;--i){
                sells[i]=Math.max(sells[i],buys[i]+x);
                buys[i]=Math.max(buys[i],(i>=1?sells[i-1]:0)-x);
            }
        }
        return sells[k-1];
    }

    //类似office的做法: tmpMax means the maximum profit of just doing at most i-1 transactions, using at most first j-1 prices, and buying the stock at price[j] - this is used for the next loop.
    public int maxProfitK(int k, int[] prices) {
        int len = prices.length;
        if (k >= len / 2) return quickSolve(prices);

        int[][] t = new int[k + 1][len];
        for (int i = 1; i <= k; i++) {
            int tmpMax =  -prices[0];
            for (int j = 1; j < len; j++) {
                t[i][j] = Math.max(t[i][j - 1], prices[j] + tmpMax);
                tmpMax =  Math.max(tmpMax, t[i - 1][j - 1] - prices[j]);
            }
        }
        return t[k][len - 1];
    }



    public int maxProfitWithCoolDown(int[] prices) {
        int n = prices.length;
        if(n<=1)
            return 0;
        int []sell=new int[n];
        int []buy = new int[n];
        buy[0]=-prices[0];
        for(int i=1;i<n;++i){
            sell[i]=Math.max(buy[i-1]+prices[i],sell[i-1]);
            buy[i]=Math.max(buy[i-1],(i>=2?sell[i-2]:0)-prices[i]);
        }
        return Math.max(sell[n-1],buy[n-1]);
    }

    public int maxProfitWithTransactionFee(int[] prices, int fee) {
        int n = prices.length;
        int []buy = new int[n];
        int []sell = new int[n];
        buy[0]=-prices[0];
        for(int i=1;i<n;++i){
            buy[i] = Math.max(buy[i-1],sell[i-1]-prices[i]);
            sell[i]=Math.max(sell[i-1],buy[i-1]+prices[i]-fee);
        }
        return Math.max(buy[n-1],sell[n-1]);
    }//you can even save space







    public String toHex(int num) {
        char[]ss={'a','b','c','d','e','f'};
        StringBuilder sb = new StringBuilder();
        if(num==0)
            return "0";
        while(num!=0){
            //System.out.println(num);
            int x = (num%16+16)%16;

            /*
            int x = num&0xf;
            int x = (num%16+16)%16;
            挺有意思的
             */


            //System.out.println(x);
            if(x>=10)
                sb.append(ss[x-10]);
            else
                sb.append(x);
            num>>>=4;
        }
        sb.reverse();
        return sb.toString();
    }


    public TreeNode rightNode(TreeNode root,TreeNode node){
        //bfs
        if(root==null||node==null)
            return null;
        Queue<TreeNode>q = new LinkedList<>();
        q.offer(root);
        TreeNode pre = null;
        while(!q.isEmpty()){
            int size = q.size();
            while(size-- >0){
                TreeNode top = q.poll();
                if(pre==node)
                    return top;
                if(top.left!=null)
                    q.offer(top.left);
                if(top.right!=null)
                    q.offer(top.right);
                pre = top;
            }
            pre=null;
        }
        return null;
    }

    public int getHeight(TreeNode root,TreeNode node){
        if(root==null)
            return Integer.MIN_VALUE;
        if(root==node)
            return 0;
        int l = getHeight(root.left,node);
        int r = getHeight(root.right,node);
        if(l==r && l==Integer.MIN_VALUE)
            return Integer.MIN_VALUE;
        return 1+Math.max(l,r);
    }
    TreeNode previous =  null;
    public TreeNode dfs(TreeNode root,TreeNode node, int level, int h){
        if(root==null)
            return null;
        if(level==h){
            if(previous==node)
                return root;
            previous = root;
        }
        TreeNode l = dfs(root.left,node,level+1,h);
        if(l!=null)
            return l;
        return dfs(root.right,node,level+1,h);
    }
    public TreeNode rightNodeDFS(TreeNode root,TreeNode node){
        if(root==null||node==null)
            return null;
        int h = getHeight(root,node);
        return dfs(root,node,0,h);
    }

    public static String bigSum(String a,String b){
        int m = a.length(), n = b.length();
        char []ss = new char[m+n];
        Arrays.fill(ss,'0');
        int i=m-1,j=n-1,carry=0,ind=m+n-1;
        while(i>=0 || j>=0 ||carry>0){
            int sum = carry + (i>=0?a.charAt(i)-'0':0)+(j>=0?b.charAt(j)-'0':0);
            ss[ind--]=(char)('0'+sum%10);
            carry=sum/10;
            if(i>=0)
                i--;
            if(j>=0)
                j--;
        }
        ind=0;
        for(;ind<m+n-1;++ind)
            if(ss[ind]!='0')
                break;
        String ans = String.valueOf(ss).substring(ind);
        if(ans.length()==a.length())
            return ans;
        else
            return a;

    }

    //k largest elements
    public static void printKlargest(){
        Scanner scanner = new Scanner(System.in);
        int T = scanner.nextInt();
        while(T-- >0){
            int N = scanner.nextInt();
            int K = scanner.nextInt();
            PriorityQueue<Integer>pq = new PriorityQueue<>();
            while(N-- >0){
                int num = scanner.nextInt();
                if(pq.size()<K)
                    pq.offer(num);
                else if(pq.peek()<num){
                    pq.poll();
                    pq.offer(num);
                }
            }
            List<Integer>ans = new ArrayList<>();
            while(!pq.isEmpty()){
                ans.add(pq.poll());
            }
            int n = ans.size();
            for(int i=n-1;i>=0;--i)
                System.out.print(ans.get(i)+" ");
        }
    }

    void countDistinct(int A[], int k, int n) {
        // Your code here
        Map<Integer,Integer>map = new HashMap<>();
        for(int i=0;i<n;++i){
            map.put(A[i],map.getOrDefault(A[i],0)+1);
            if(i>=k-1){
                System.out.print(map.size()+" ");
                map.put(A[i-k+1],map.get(A[i-k+1])-1);
                if(map.get(A[i-k+1])==0)
                    map.remove(A[i-k+1]);
            }
        }
    }



    public static void insertSort(int[]arr){
        int n = arr.length;
        int negative = 0;
        for(int x:arr)
            if(x<0)
                negative++;
        for(int i=0;i<negative;++i){
            if(arr[i]<0)
                continue;
            int j=i+1;
            for(;j<n;++j){
                if(arr[j]<0){
                    break;
                }
            }
            //right shift the array by one
            int save = arr[j];
            for(int k=j;k>=i+1;k--)
                arr[k]=arr[k-1];
            arr[i]=save;
        }
        for(int x:arr)
            System.out.println(x);
    }

    //URLify a given string
    public static void URLify(String str){
        str.trim();
        StringBuilder sb = new StringBuilder();
        int n = str.length();
        for(int i=0;i<n;++i){
            if(str.charAt(i)!=' ')
                sb.append(str.charAt(i));
            else
                sb.append("%20");
        }
        System.out.println( sb.toString());
    }

    public static int getDifficulty(String str){
        str = str.trim();
        String []strs = str.split("\\s+");
        Set<Character>set = new HashSet<>();
        char []owe = {'a','e','i','o','u'};
        for(char c:owe)
            set.add(c);
        int diff = 0;
        int easy=0;
        for(String ele:strs){
            if(ele.isEmpty()||ele.equals(" "))
                continue;
            char []ss = ele.toCharArray();
            int num = 0;
            int owels=0;
            for(char c:ss){
                if(set.contains(Character.toLowerCase(c))){
                    owels++;
                    if(num>=4)
                        break;
                    num=0;
                }else{
                    num++;
                    if(num>=4)
                        break;
                }
            }
            if(num>=4||ss.length-owels>owels){
                diff++;
            }else
                easy++;
        }
        return 5*diff+3*easy;
    }


    public static void transform(int[]arr){
        List<Integer>ans = new ArrayList<>();
        for(int x:arr){
            if(x==0)
                continue;
            if(ans.isEmpty()){
                ans.add(x);
                continue;
            }
            if(ans.get(ans.size()-1)==x){
                ans.set(ans.size()-1,2*x);
            }else
                ans.add(x);
        }
        int n = arr.length,m=ans.size();
        for(int i=0;i<n;++i){
            if(i<m)
                System.out.print(ans.get(i)+" ");
            else
                System.out.print(0+" ");
        }

    }

    public static void stringComparasion(String a, String b){
        if(a.equals(b)){
            System.out.println(0);
            return;
        }
        int m = a.length(), n= b.length();
        int i=0,j=0;
        while(i<m && j<n){
            if(a.charAt(i)!=b.charAt(j)){
                System.out.println(a.charAt(i)>b.charAt(j)?1:-1);
                return;
            }else if(a.charAt(i)=='n'){
                if(i<m-1 && j<n-1){
                    if(a.charAt(i+1)=='g' && b.charAt(j+1)=='g'){
                        i+=2;
                        j+=2;
                    }else if(a.charAt(i+1)!='g' && b.charAt(j+1)!='g'){
                        i++;
                        j++;
                    }else if(a.charAt(i+1)=='g'||b.charAt(j+1)=='g'){
                        System.out.println(b.charAt(j+1)=='g'?-1:1);
                        return;
                    }
                }else if(i==m-1 && j==n-1){
                    System.out.println(0);
                    return;
                }else if(i==m-1 || j==n-1){
                    System.out.println(i==m-1?-1:1);
                    return;
                }
            }else{
                i++;
                j++;
            }
        }
        System.out.println(i==m?-1:1);
    }


    //Find K Pairs with Smallest Sums
    public static void findKthInTwoSum(int[]arr1,int []arr2,int k){
        PriorityQueue<int[]>pq=new PriorityQueue<>(new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return arr1[o1[0]]+arr2[o1[1]]-arr1[o2[0]]-arr2[o2[1]];
            }
        });
        int m = arr1.length, n = arr2.length;
        for(int i=0;i<n;++i)
            pq.offer(new int[]{0,i});
        Set<Integer>set = new HashSet<>();
        while(set.size()<k){
            int []top = pq.poll();
            set.add(arr1[top[0]]+arr2[top[1]]);
            if(set.size()==k)
                System.out.println(arr1[top[0]]+arr2[top[1]]);
            if(top[0]<m-1)
                pq.offer(new int[]{top[0]+1,top[1]});
        }
    }

    public static void ignoreString(String str){
        Map<Character,Integer>map = new HashMap<>();
        char []ss = str.toCharArray();
        StringBuilder sb = new StringBuilder();
        for(char c:ss){
            char cc = Character.toLowerCase(c);
            if(!map.containsKey(cc))
                map.put(cc,0);
            map.put(cc,map.get(cc)+1);
            if((map.get(cc)&0x1)!=0)
                sb.append(c);
        }
        System.out.println(sb.toString());
    }


    public static void countOfCarry(String a, String b){
        int m = a.length(), n = b.length();
        char []ss = new char[m+n];
        int ind = m+n-1,i=m-1,j=n-1,carry=0,cnt=0;
        while(i>=0 || j>=0 || carry>0){
            int sum = carry+ (i>=0?a.charAt(i)-'0':0)+(j>=0?b.charAt(j)-'0':0);
            if(sum>=10)
                cnt++;
            ss[ind--]=(char)(sum%10+'0');
            carry=sum/10;
        }
        System.out.println(cnt);

    }

    public static void rottenOrange(int[][]matrix){
        int m = matrix.length, n = matrix[0].length;
        int fresh=0;
        Queue<int[]>q = new LinkedList<>();
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(matrix[i][j]==2)
                    q.offer(new int[]{i,j});
                if(matrix[i][j]==1)
                    fresh++;
            }
        }
        int[]dx = {1,-1,0,0};
        int[]dy={0,0,1,-1};
        int time=0;
        while(!q.isEmpty()){
            int size = q.size();
            while(size-- >0){
                int []top = q.poll();
                for(int k=0;k<4;++k){
                    int nx = top[0]+dx[k];
                    int ny = top[1]+dy[k];
                    if(nx>=0 && nx<m && ny>=0 && ny<n && matrix[nx][ny]==1){
                        q.offer(new int[]{nx,ny});
                        matrix[nx][ny]=2;
                        fresh--;
                    }
                }
            }
            time++;
        }
        System.out.println(fresh==0?time:-1);
    }


    //total count of sub-arrays
    public static void countOfSubArrays(int[]nums){
        Map<Integer,Integer>map = new HashMap<>();
        map.put(0,1);//sentinel
        int n = nums.length;
        int sum=0,cnt=0;
        for(int i=0;i<n;++i){
            sum+=nums[i];
            if(map.containsKey(sum)){
                cnt+=map.get(sum);
            }
            map.put(sum,map.getOrDefault(sum,0)+1);
        }
        System.out.println(cnt);

    }

    public static void findTwoNumber(int[]arr){
        int n = arr.length;
        int xor = 0;
        for(int x:arr){
            xor^=x;
        }
        //xor&=(-xor);//find the first different bit
        xor -=(xor&(xor-1));
        int a = 0,b=0;
        for(int x:arr){
            if((x&xor)!=0){
                a^=x;
            }else
                b^=x;
        }
        System.out.println(a<=b?a+" "+b:b+" "+a);

    }


    public static void dfs(int x,int y,int n,char[][]matrix,int[]ans,int[]dx,int[]dy){
        if(n==1){
            ans[0]++;
            return;
        }
        for(int k=0;k<5;++k){
            int nx = x+dx[k];
            int ny = y+dy[k];
            if(nx>=0 && nx<4 && ny>=0 &&  ny<3  && Character.isDigit(matrix[nx][ny])){
                dfs(nx,ny,n-1,matrix,ans,dx,dy);
            }
        }

    }
    public static void numberOfPhone(int n){
        char[][]matrix= {{'1','2','3'},{'4','5','6'},{'7','8','9'},{'*','0','#'}};
        int []ans={0};
        int[]dx={1,-1,0,0,0};
        int[]dy ={0,0,1,-1,0};

        for(int i=0;i<=8;++i){

            dfs(i/3,i%3,n,matrix,ans,dx,dy);
        }
        dfs(3,1,n,matrix,ans,dx,dy);
        System.out.println(ans[0]);
    }

    public static String removeDup(String word) {
        int checker = 0;
        StringBuilder builder = new StringBuilder();
        for(char c : word.toCharArray()) {
            int num = c;
            if ((checker & (1 << num)) == 0) builder.append(c);
            checker |= (1 << num);
        }
        return builder.toString();
    }

    //Stock buy and sell
    //很经典
    public static void buyAndSell(int[]nums){
        int n = nums.length;
        for(int i=n-1;i>=1;--i){
            nums[i]-=nums[i-1];
        }
        nums[0]=0;
        int start=0,ind=0;
        boolean has=false;
        int sum=0;
        for(ind=1;ind<=n;++ind){
            if(ind<n && nums[ind]>=0)
                sum+=nums[ind];
            if(ind==n||nums[ind]<0){
                if(start!=ind-1 && sum>0){
                    has=true;
                    System.out.print("("+start+" " +(ind-1)+") ");
                }
                start=ind;
                sum=0;
            }
        }
        if(!has){
            System.out.print("No Profit");
        }
        System.out.println();
    }

    public static void majorityElement(int[]nums){
        int n = nums.length;
        int a=0,cnt=0;
        for(int x:nums){
            if(x==a){
                cnt++;
            }else if(a==0){
                cnt=1;
                a=x;
            }else
                cnt--;
        }
        //check the number
        cnt=0;
        for(int x:nums){
            if(a==x)
                cnt++;
        }
        System.out.println(cnt>n/2?a:"NO Majority Element");
    }

    public static void swap1(int[]nums,int begin,int end){
        int tmp = nums[begin];
        nums[begin]=nums[end];
        nums[end]=tmp;
    }
    public static void sortColors(int[]nums){
        int begin=0,n=nums.length,end=n-1,mid=0;
        while(mid<=end){
            if(nums[mid]==0){
                swap1(nums,mid,begin);
                mid++;
                begin++;
            }else if(nums[mid]==1){
                mid++;
            }else{
                swap1(nums,mid,end);
                end--;
            }
        }
        for(int x:nums){
            System.out.println(x);
        }
    }

    public static void printMerge(int[]a,int[]b){
        int m = a.length, n = b.length;
        if(a[m-1]>a[0])
            reverse(a);
        if(b[n-1]>b[0])
            reverse(b);

    }

    public static void getNumOfAnagram(String a, String b){
        char []bb = b.toCharArray();
        Arrays.sort(bb);
        b = String.valueOf(bb);
        int m =b.length(),n=a.length();
        int cnt=0;
        for(int i=0;i<=n-m;++i){
            String tmp = a.substring(i,i+m);
            char []ss = tmp.toCharArray();
            Arrays.sort(ss);
            if(String.valueOf(ss).equals(b)){
                cnt++;
            }
        }
        System.out.println(cnt);
    }

    public static void simplebackpack(int weight,int[]values,int[]weights){
        int []dp = new int[weight+1];
        int n = values.length;
        for(int i=0;i<n;++i){
            for(int val = weight;val>=weights[i];--val){
                dp[val]=Math.max(dp[val],dp[val-weights[i]]+values[i]);
            }
        }
        System.out.println(dp[weight]);
    }

    public static void addBinary(String a, String b){
        int carry=0, m = a.length(), n= b.length();
        int i=m-1,j=n-1;
        StringBuilder sb = new StringBuilder();
        while(i>=0 || j>=0|| carry>0){
            int sum = carry + (i>=0?a.charAt(i--)-'0':0)+(j>=0?b.charAt(j--)-'0':0);
            sb.append(sum%2);
            carry=sum/2;
        }
        sb.reverse();
        System.out.println(sb.toString());
    }

    public static int cnt =0;
    public  static void merge(int[]nums,int begin,int mid,int end){
        int []ans = new int[end-begin+1];
        int ind=0,i=begin,j=mid+1;
        while(i<=mid && j<=end){
            if(nums[j]<nums[i]){
                cnt+=(mid-i+1);
                ans[ind++]=nums[j++];
            }else{
                ans[ind++]=nums[i++];
            }
        }
        while(i<=mid)
            ans[ind++]=nums[i++];
        while(j<=end)
            ans[ind++]=nums[j++];
        for(int ii=0;ii<end-begin+1;++ii){
            nums[ii+begin]=ans[ii];
        }
    }
    public static void  mergeSort(int[]nums,int begin,int end){
        if(begin>=end){
            return;
        }
        int mid = (end-begin)/2+begin;
        mergeSort(nums,begin,mid);
        mergeSort(nums,mid+1,end);
        merge(nums,begin,mid,end);
    }

    public static TreeNode bstToDll(TreeNode root){
        TreeNode dummy = new TreeNode(0);
        TreeNode prev11 = dummy;
        prev11=helper(root,prev11);
        dummy.right.left = null;
        System.out.println(dummy.right.val);
        return dummy.right;
    }

    public static TreeNode helper(TreeNode root,TreeNode prev11){
        if(root==null)
            return prev11;
        prev11=helper(root.left,prev11);
        root.left = prev11;
        prev11.right = root;
        prev11 = root;
        prev11=helper(root.right,prev11);
        return prev11;
    }


    public static void print(int a ,int b,boolean[]turn){
        if(b==a){
            System.out.print(a+" ");
            return;
        }
        System.out.print(b+" ");
        if(b<0){
            turn[0]=true;
        }
        if(!turn[0]){
            print(a,b-5,turn);
        }else{
            print(a,b+5,turn);
        }
    }



    //Two numbers with sum closest to zero
    public static void cloestToZero(int[]nums){
        int n = nums.length;
        if(n==1){
            System.out.print(nums[0]+" "+nums[0]);
        }
        TreeSet<Integer>set = new TreeSet<>();
        for(int x:nums){

        }

    }

    //Jumping Numbers
    public static List<Integer> jumpingNumbers(int n){
        Queue<Integer>q = new LinkedList<>();
        for(int i=0;i<=9;++i)
            q.offer(i);
        Set<Integer>ans = new HashSet<>();
        PriorityQueue<Integer> pq = new PriorityQueue<Integer>(new Comparator<Integer>(){
            public int compare(Integer a, Integer b){
                String aa = String.valueOf(a);
                String bb = String.valueOf(b);
                int rs = (int)aa.charAt(0) - (int)bb.charAt(0);
                if(rs == 0){
                    if(aa.length() != bb.length()){
                        return aa.length() - bb.length();
                    }else{
                        return a.compareTo(b);
                    }
                }else{
                    return rs;
                }
            }
        });
        while(!q.isEmpty()){
            int top = q.poll();
            if(!ans.contains(top)){
                pq.offer(top);
                ans.add(top);
            }
            if(top%10!=9){
                int top1 = 10*top+top%10+1;
                if(top1<=n)
                    q.offer(top1);
            }
            if(top%10!=0){
                int top2 = 10*top+top%10-1;
                if(top2<=n)
                    q.offer(top2);
            }
        }
        List<Integer>anss = new ArrayList<>();
        while(!pq.isEmpty())
            anss.add(pq.poll());
        return anss;
    }//

    public static void dfs(int num,int target,Set<Integer>vis){
        if(num<=target && !vis.contains(num)){
            System.out.print(num+" ");
            vis.add(num);
        }
        if(num%10!=0){
            int top2 = 10*num+num%10-1;
            if(top2<=target)
                dfs(top2,target,vis);
        }
        if(num%10!=9){
            int top1 = 10*num+num%10+1;
            if(top1<=target)
                dfs(top1,target,vis);
        }

    }
    public static void jumpingNumbersDFS(int target){
        Set<Integer>vis = new HashSet<>();
        for(int i=0;i<=9;++i){
            if(i<=target){
                dfs(i,target,vis);
            }
        }
    }

    public Node flatten(Node root)
    {
        // Your code here
        PriorityQueue<Node>pq = new PriorityQueue<>(new Comparator<Node>() {
            @Override
            public int compare(Node o1, Node o2) {
                return o1.data-o2.data;
            }
        });
        while(root!=null){
            pq.offer(root);
            root=root.next;
        }
        Node dummy = new Node(0);
        Node p = dummy;
        while(!pq.isEmpty()){
            Node top = pq.poll();
            p.next = top;
            p=p.next;
            if(top.bottom!=null)
                pq.offer(top.bottom);
        }
        return dummy.next;

    }

    public static String multiply(String num1, String num2) {
        int m = num1.length(), n = num2.length();
        char []ss = new char[m+n];
        Arrays.fill(ss,'0');
        for(int i=m-1;i>=0;--i){
            int carry=0;
            for(int j=n-1;j>=0;--j){
                int sum = carry+(ss[i+j+1]-'0')+(num1.charAt(i)-'0')*(num2.charAt(j)-'0');
                ss[i+j+1] = (char)(sum%10+'0');
                carry=sum/10;
            }
            ss[i]+= carry;
        }
        int ind=0;
        while(ind<m+n-1){
            if(ss[ind]!='0')
                break;
            ind++;
        }
        return String.valueOf(ss).substring(ind);
    }

    static class ArrayContainer{
        public int[]nums;
        public int index;
        public ArrayContainer(int[]nums,int index){
            this.nums = nums;
            this.index = index;
        }
    }

    public static ArrayList<Integer> mergeKArrays(int[][] arrays,int k)
    {
        //add code here.
        PriorityQueue<ArrayContainer>pq = new PriorityQueue<>(new Comparator<ArrayContainer>() {
            @Override
            public int compare(ArrayContainer o1, ArrayContainer o2) {
                return o1.nums[o1.index]-o2.nums[o2.index];
            }
        });
        int m = arrays.length;
        for(int i=0;i<m;++i){
            pq.offer(new ArrayContainer(arrays[i],0));
        }
        ArrayList<Integer>ans = new ArrayList<>();
        while(!pq.isEmpty()){
            ArrayContainer top = pq.poll();
            ans.add(top.nums[top.index]);
            if(top.index+1<top.nums.length){
                pq.offer(new ArrayContainer(top.nums,top.index+1));
            }
        }
        return ans;

    }


    //Longest Common Substring
    public static int longestCommonSubStr(String str1,String str2){
        int m = str1.length(),n=str2.length();
        int[][]dp = new int[m+1][n+1];
        int ans =0;
        for(int i=1;i<=m;++i){
            for(int j=1;j<=n;++j){
                if(str1.charAt(i-1)==str2.charAt(j-1)){
                    dp[i][j]=dp[i-1][j-1]+1;
                }else
                    dp[i][j]=0;
                ans=Math.max(ans,dp[i][j]);
            }
        }
        return ans;
    }

    public static int quickSelect(int[]nums,int begin,int end){
        int low = begin, hi =end,key=nums[low];
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
    public static int findkth(int[]nums,int begin,int end,int k){
        int ind = quickSelect(nums,begin,end);
        if(ind==k-1)
            return nums[ind];
        else if(ind>k-1)
            return findkth(nums,begin,ind-1,k);
        else
            return findkth(nums,ind+1,end,k);
    }

    public static TreeNode inorderSuccessor1(TreeNode root, TreeNode node){
        if(node==null||root==null)
            return null;
        if(node.val>=root.val)
            return inorderSuccessor1(root.right,node);
        else{
            TreeNode l = inorderSuccessor1(root.left,node);
            return l!=null?l:root;
        }
    }
    public static TreeNode findSmallest(TreeNode root){
        TreeNode p = root;
        while(p.left!=null){
            p=p.left;
        }
        return p;
    }
    public static void merge(TreeNode root1,TreeNode root2){
        TreeNode p = findSmallest(root1);
        TreeNode q = findSmallest(root2);
        System.out.println(p.val);
        System.out.println(q.val);
        int val =0;
        while(p!=null||q!=null){
            if(p==null){
                val=q.val;
                q=inorderSuccessor1(root2,q);
            }else if(q==null){
                val = p.val;
                p=inorderSuccessor1(root1,p);
            }else{
                if(p.val>q.val){
                    val = q.val;
                    q=inorderSuccessor1(root2,q);
                }else{
                    val=p.val;
                    p=inorderSuccessor1(root1,p);
                }
            }
            System.out.print(val+" ");
        }
    }

    public static void dfs(int[]nums,int target,int ind,List<Integer>path){
        if(target==0){
            System.out.print("(");
            int n = path.size();
            for(int i=0;i<n-1;++i){
                System.out.print(path.get(i)+" ");
            }
            System.out.print(path.get(n-1)+")");
        }
        for(int i=ind;i<nums.length;++i){
            if(target>=nums[i]){
                path.add(nums[i]);
                dfs(nums,target-nums[i],i,path);
                path.remove(path.size()-1);
            }else
                break;
        }
    }

    public static void combinationSum(int[]nums,int target){
        int n = nums.length;
        Arrays.sort(nums);
        ArrayList<Integer>path = new ArrayList<>();
        dfs(nums,target,0,path);
        System.out.println();
    }

    //Is Binary Number Multiple of 3
    //3 的和有可能被整除
    //2^n %3==2  when n is odd, otherwise it is 1
    public static boolean isMutileThree(String binary){
        //binary
        int n = binary.length();
        int mod=0;
        for(int i=n-1;i>=0;--i){
            if(binary.charAt(i)=='1'){
                if((n-i-1)%2==0)
                    mod++;
                else
                    mod+=2;
            }
        }
        return mod%3==0;
    }
    //subset
    public static void dfs(int[]nums,int ind,List<Integer>path){
        if(path.isEmpty()){
            System.out.print("()");
        }else{
            System.out.print("(");
            int n= path.size();
            for(int i=0;i<n-1;++i){
                System.out.print(path.get(i)+" ");
            }
            System.out.print(path.get(n-1)+")");
        }
        for(int i=ind;i<nums.length;++i){
            if(i>ind && nums[i]==nums[i-1])
                continue;
            path.add(nums[i]);
            dfs(nums,i+1,path);
            path.remove(path.size()-1);
        }
    }
    public static void subsets(int[]nums){
        int n = nums.length;
        List<Integer>path=new ArrayList<>();
        Arrays.sort(nums);
        dfs(nums,0,path);
        System.out.println();
    }

    public static boolean isInterLeave(String a,String b,String c) {
        int m= a.length(), n=b.length();
        if(m+n!=c.length())
            return false;
        boolean[][]dp=new boolean[m+1][n+1];
        dp[0][0] = true;
        //deal with c
        for(int i=1;i<=m;++i)
            dp[i][0]=dp[i-1][0] && a.charAt(i-1)==c.charAt(i-1);
        for(int i=1;i<=n;++i)
            dp[0][i]=dp[0][i-1] && b.charAt(i-1)==c.charAt(i-1);
        for(int i=1;i<=m;++i){
            for(int j=1;j<=n;++j){
                if(a.charAt(i-1)==c.charAt(i+j-1))
                    dp[i][j]|=dp[i-1][j];
                if(b.charAt(j-1)==c.charAt(i+j-1))
                    dp[i][j]|=dp[i][j-1];
            }
        }
        return dp[m][n];
    }


    //Matrix Chain Multiplication
    //like the brust ballroon// think a lot



    //很有意思，值得一做
    
    public static void firstNoRepeat(String str1){
        Queue<String>q = new LinkedList<>();
        int n = str1.length();
        int []cnt=new int[26];
        for(int i=0;i<n;++i){
            String c = String.valueOf(str1.charAt(i)) ;
            cnt[c.charAt(0)-'a']++;
            if(q.isEmpty()){
                if(cnt[c.charAt(0)-'a']==1){
                    q.offer(c);
                    System.out.print(c+" ");
                }else
                    System.out.print(-1+" ");
            }else if(c.equals(q.peek())){
                q.poll();
                while(!q.isEmpty() && cnt[q.peek().charAt(0)-'a']>1){
                    q.poll();
                }
                if(q.isEmpty())
                    System.out.print(-1+" ");
                else
                    System.out.print(q.peek()+" ");
            }else{
                q.offer(c);
                System.out.print(q.peek()+" ");
            }
        }
        System.out.println();
    }

    public int trap(int[] height) {
        int n = height.length;
        if(n<=1)
            return 0;
        int left=0,leftH=height[0],right=n-1,rightH=height[n-1];
        int ans = 0;
        while(left<right){
            leftH=Math.max(leftH,height[left]);
            rightH=Math.max(rightH,height[right]);
            if(leftH<rightH){
                ans+=Math.max(leftH-height[left++],0);
            }else{
                ans+=Math.max(rightH-height[right--],0);
            }
        }
        return ans;
    }

    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        Stack<Integer>stk = new Stack<>();
        int ans=0;
        for(int i=0;i<=n;++i){
            int h = (i==n?0:heights[i]);
            while(!stk.isEmpty() && heights[stk.peek()]>h){
                int upper = heights[stk.pop()];
                int len = stk.isEmpty()?i:i-stk.peek()-1;
                ans=Math.max(ans,upper*len);
            }
            stk.push(i);
        }
        return ans;
    }
    public int maxArea(int a[][],int m,int n){
        int []dp = new int[n];
        int ans = 0;
        for(int j=0;j<n;++j){
            dp[j]= a[0][j]==0?0:1;
            ans=Math.max(ans,dp[j]);
        }
        for(int i=1;i<m;++i){
            for(int j=0;j<n;++j){
                if(a[i][j]!=0)
                    dp[j]=1+dp[j];
                else
                    dp[j]=0;
            }
            ans=Math.max(ans,largestRectangleArea(dp));
        }
        return ans;
    }

    //Longest Even Length Substring
    //A O(n^2) time and O(1) extra space solution: start from middle, 然后拓展
    public static int longestEvenSubstr(String str){
        char []ss = str.toCharArray();
        int n = ss.length;
        int []sums = new int[n];
        for(int i=0;i<n;++i){
            sums[i]=(i==0?0:sums[i-1])+ss[i]-'0';
        }
        int len =0;
        for(int i=1;i<n;++i){
            for(int j=i-1;j>=0;--j){
                if((i-j+1)%2==0){
                    int leng = (i-j+1)/2;
                    int low = j<=0?0:sums[j-1];
                    if(sums[i]-sums[i-leng]==sums[i-leng]-low){
                        len=Math.max(2*leng,len);
                    }
                }
            }
        }
        return len;
    }

    /*

    int countCommon(Node *a, Node *b)
{
    int count = 0;

    // loop to count coomon in the list starting
    // from node a and b
    for (; a && b; a = a->next, b = b->next)

        // increment the count for same values
        if (a->data == b->data)
            ++count;
        else
            break;

    return count;
}

    int maxPalindrome(Node *head)
{
    int result = 0;
    Node *prev = NULL, *curr = head;

    // loop till the end of the linked list
    while (curr)
    {
        // The sublist from head to current
        // reversed.
        Node *next = curr->next;
        curr->next = prev;

        // check for odd length palindrome
        // by finding longest common list elements
        // beginning from prev and from next (We
        // exclude curr)
        result = max(result,
                     2*countCommon(prev, next)+1);

        // check for even length palindrome
        // by finding longest common list elements
        // beginning from curr and from next
        result = max(result,
                     2*countCommon(curr, next));

        // update prev and curr for next iteration
        prev = curr;
        curr = next;
    }
    return result;
}
     */

    public int canCompleteCircuit(int[] gas, int[] cost) {
        int n = gas.length;
        int ans = 0;
        int cur=0,start=0;
        for(int i=0;i<n;++i){
            ans+=(gas[i]-cost[i]);
            cur+=(gas[i]-cost[i]);
            if(cur<0){
                cur=0;
                start=i+1;
            }
        }

        return ans>=0?start:-1;
    }


    //relative sort
    public static void relativeSort(int[]A,int[]B){
        int m = A.length, n = B.length;
        Map<Integer,Integer>map = new HashMap<>();
        for(int x:B){
            map.put(x,0);
        }
        List<Integer>ans = new ArrayList<>();
        for(int x:A){
            if(map.containsKey(x)){
                map.put(x,map.get(x)+1);
            }else{
                ans.add(x);
            }
        }
        for(int x:B){
            int nn = map.get(x);
            while(nn-- >0){
                System.out.print(x+" ");
            }
        }
        Collections.sort(ans);
        for(int x:ans){
            System.out.print(x+" ");
        }
    }


    //Factorials of large numbers



    public static int maxCoins(int[] nums) {
        int n = nums.length;
        int [][]dp = new int[n+2][n+2];
        int []copy = new int[n+2];
        copy[0]=copy[n+1]=1;
        for(int i=1;i<=n;++i)
            copy[i]=nums[i-1];
        for(int len =2;len<=n+1;len++){
            for(int left = 0;left<=n+1-len;++left){
                int right = left+len;
                for(int k=left+1;k<right;++k){
                    dp[left][right]=Math.max(dp[left][right],copy[k]*copy[left]*copy[right]+dp[left][k]+dp[k][right]);
                }
            }
        }
        return dp[0][n+1];
    }



    //Matrix Chain Multiplication
    public static String print(int[][]bracket,int begin,int end){
        int parent = bracket[begin][end];
        if(begin==end-1)
            return ""+(char)(begin+'A');
        if(parent==begin+1 && parent==end-1)
           return "("+(char)(begin+'A')+""+(char)(parent+'A')+")";
        if(parent==end-1){
            return "("+print(bracket,begin,parent)+(char)(parent+'A')+")";
        }else if(parent==begin+1)
            return "("+(char)(begin+'A')+print(bracket,parent,end)+")";
        else
            return "("+print(bracket,begin,parent)+print(bracket,parent,end)+")";
    }
    public static int matrixChain(int[]nums){
        int n = nums.length;
        int [][]dp = new int[n][n];
        int [][]bracket = new int[n][n];
        for(int len=2;len<=n-1;++len){
            for(int left=0;left<=n-1-len;++left){
                int right = left+len;
                dp[left][right]=2147483647;
                int kk=0;
                for(int k=left+1;k<right;++k){
                    if(dp[left][right]>nums[k]*nums[left]*nums[right]+dp[left][k]+dp[k][right]){
                        kk=k;
                        dp[left][right]=nums[k]*nums[left]*nums[right]+dp[left][k]+dp[k][right];
                    }
                }
                //System.out.println(left+" "+kk+" "+right);
                bracket[left][right]=kk;
            }
        }
        System.out.println(print(bracket,0,n-1));
        return dp[0][n-1];
    }

    //Consecutive 1's not allowed
    //dp[n][0]--start with 0
    //dp[n][1]--start with 1
    public static void numberOfConsecutive(int n){
        int [][]dp = new int[n+1][2];
        dp[1][0]=dp[1][1]=1;
        dp[2][0]=2;
        dp[2][1]=1;
        int mod =1000000007;
        for(int i=3;i<=n;++i){
            dp[i][0]=((2*dp[i-2][0])%mod+dp[i-2][1]%mod)%mod;
            dp[i][1]=(dp[i-2][0]%mod+dp[i-2][1]%mod)%mod;
        }
        System.out.println((dp[n][0]%mod+dp[n][1]%mod)%mod);
    }

    public static void shortUrl(int n){
        String map = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        char[]chrs = map.toCharArray();
        int save =n;
        StringBuilder sb = new StringBuilder();
        while(n!=0){
            sb.append(chrs[n%62]);
            n/=62;
        }
        sb.reverse();
        System.out.println(sb.toString());
        System.out.println(save);
    }


    //stack box
    /*

class Box:
    def __init__(self,h,w,l):
        self.h=h
        self.w=w
        self.l=l
    def __str__(self):
        return str(self.l)+" "+str(self.w)+" "+str(self.h)

##assume l is always larger or equal to w
def maxHeight(height, width, length, n):
    boxs=[]
    for i in range(0,n):
        boxs.append(Box(height[i],min(width[i],length[i]),max(width[i],length[i]))) ##attention
        boxs.append(Box(width[i],min(height[i],length[i]),max(height[i],length[i])))
        boxs.append(Box(length[i],min(height[i],width[i]),max(height[i],width[i])))
    boxs.sort(key=lambda box: box.w*box.l, reverse=True)
    # for x in boxs:
    #     print(x)
    dp=[0]*(3*n)
    ans=0
    for i in range(0,3*n):
        dp[i]=max(dp[i],boxs[i].h)
        for j in range(i-1,-1,-1):
            if boxs[j].w>boxs[i].w and boxs[j].l>boxs[i].l:
                dp[i]=max(dp[i],dp[j]+boxs[i].h)
        ans = max(ans, dp[i])
    return ans
     */

    public static String toBinary(int x,int n){
        StringBuilder sb = new StringBuilder();
        while(x!=0){
            sb.append(x%2);
            x/=2;
        }
        for(int i=sb.length();i<n;++i)
            sb.append('0');
        sb.reverse();
        return sb.toString();
    }
    public static void generateCode(int n) {
        // Your code here
        List<Integer>ans = new ArrayList<>();
        ans.add(0);
        ans.add(1);
        for(int i=1;i<n;++i){
            int nn = ans.size();
            for(int j=nn-1;j>=0;--j){
                ans.add((ans.get(j)|1<<i));
            }
        }
        for(int x:ans){
            System.out.print(toBinary(x,n)+" ");
        }
    }
    public static void main(String[]args){
        MicrosoftOnsite mo = new MicrosoftOnsite();
        TreeNode root = new TreeNode(6);
        root.left = new TreeNode(1);
        root.right = new TreeNode(7);
        root.left.left = new TreeNode(2);
        root.left.right = new TreeNode(4);
        root.right.right =new TreeNode(8);
        root.right.left =new TreeNode(6);
        TreeNode root2 = new TreeNode(4);
        root2.left = new TreeNode(1);
        root2.right=new TreeNode(33);
        //merge(root,root2);
        //bstToDll(root);
        //reverseLevelOrder(root);
//        root.left.right.left =new TreeNode(7);
//        root.left.right.right = new TreeNode(4);
        //System.out.println(mo.lowestCommonAncestorAnotherWay(root,root.left,root.left.right.right).val);
       // System.out.println(mo.inorderSuccessorIteratvie(root,root.right.left).val);

//        TreeLinkNode node = new TreeLinkNode(1);
//        node.left = new TreeLinkNode(2);
//        node.right = new TreeLinkNode(3);
//        node.left.left = new TreeLinkNode(4);
//        node.left.right = new TreeLinkNode(5);
//        node.right.right = new TreeLinkNode(7);
//        mo.connect(node);
//        int []nums={-1,-1,0,0,2};
//        mo.permuteUnique(nums);
//
//        LRUCache cache = new LRUCache( 2 /* capacity */ );
//
//        cache.put(2, 1);
//        cache.put(2, 2);
//        System.out.println(cache.get(2));       // returns 1
//        cache.put(1, 1);    // evicts key 2
//        cache.put(4, 1);    // evicts key 1
//        System.out.println(cache.get(2));       // returns -1 (not found)

//        int []nums={1,2,3,4,5,6,7};
//        mo.rotateByGCD(nums,3);
        //System.out.println(mo.strStrKMP("aaaaa","bba"));
        //System.out.println(mo.rightNodeDFS(root,root));
        //System.out.println(bigSum("45678990567890","563456789"));
//        int []nums ={2, -4, 7, -3, 4};
//        insertSort(nums);
        //int []nums={2, 4, 5, 0, 0, 5, 4, 8, 6, 0, 6, 8};
        //stringComparasion("wnngxqnldwi", "wnngxqngldww");
//        int []arr1={1, 3, 4, 8, 10};
//        int []arr2={20, 22, 30, 40,};
//        findKthInTwoSum(arr1,arr2,4);
        //ignoreString("It is a long day dear.");
        //countOfCarry("2465","535");
        //int []nums={2,1,3,2,1,4};
        //countOfSubArrays(nums);
        //findTwoNumber(nums);
        //numberOfPhone(13);//会爆
        //System.out.println(removeDup("geeks for@@geeks"));
        //int[]nums={23,13,25,29,33,19,34,45,65,67};
        //int []nums={886,2777,6915,7793,8335,5386,492,6649,1421,2362,27,8690,59,7763,3926,540,3426,9172,5736,5211,5368,2567,6429,5782,1530,2862,5123,4067,3135,3929,9802,4022,3058,3069,8167,1393,8456,5011,8042,6229,7373,4421,4919,3784,8537,5198,4324,8315,4370,6413,3526,6091,8980,9956,1873,6862,9170,6996,7281,2305,925,7084,6327,336,6505,846,1729,1313,5857,6124,3895,9582,545,8814,3367,5434,364,4043,3750,1087,6808,7276,7178,5788};
        //output
        /*
        (0 4) (6 7) (8 9) (10 11) (12 13) (15 17) (19 20) (21 22) (24 26) (28 30) (32 34) (35 36) (37 38) (39 40) (41 42) (43 44) (46 47) (48 49) (50 53) (54 56) (57 58) (60 61) (63 64) (65 66) (67 69) (70 71) (72 73) (74 75) (76 77) (79 81)
         */
        //buyAndSell(nums);
//        int []nums={0,2,1,0,2};
//        sortColors(nums);
        //getNumOfAnagram("forf","for");
//        int []weights = {4,5,1};
//        int []values = {1,2,3};
//        simplebackpack(4,values,weights);
        //addBinary("1101","111");
//        int []nums={2,4,1,3,5};
//        mergeSort(nums,0,4);
//        System.out.println(cnt);
        //boolean []turn={false};
        //print(16,11,turn);
        //jumpingNumbersDFS(50);
        //System.out.println(multiply("0","0"));
        //System.out.println(longestCommonSubStr("GeeksforGeeks","GeeksQuiz"));
//        int []nums={7, 10, 4, 3, 20, 15};
//        for(int k=1;k<=nums.length;++k)
//            System.out.println(findkth(nums,0,nums.length-1,k));

        int []nums={387, 278, 416, 294, 336, 387, 493, 150, 422, 363, 28, 191, 60, 264};
        //combinationSum(nums,16);
        //subsets(nums);

//        System.out.println(isInterLeave("YX","X","XXY"));
//        System.out.println(isInterLeave("XY","X","XXY"));
//        ListNode node = new ListNode(1);
//        node.next = new ListNode(3);
//        System.out.println(getIntersectionNode1(node,node.next).val);
        //System.out.println(longestEvenSubstr("1234123"));
        //System.out.println(maxCoins(nums));
        //System.out.println(matrixChain(nums));
        //numberOfConsecutive(100);
        //shortUrl(12345);
        generateCode(2);

    }
}
