import commons.*;

import java.util.*;

/**
 * Created by tao on 10/28/17.
 */


//encode and decode string
class CodecString {

    // Encodes a list of strings to a single string.
    public String encode(List<String> strs) {
        StringBuilder sb = new StringBuilder();
        for(String str:strs){
            sb.append(str.length()).append('@').append(str);
        }
        return sb.toString();
    }

    // Decodes a single string to a list of strings.
    public List<String> decode(String s) {
        List<String>res=new ArrayList<>();
        int n = s.length(),i=0;
        int start=0;
        while(i<n){
            start=i;
            while(i<n && Character.isDigit(s.charAt(i))&&s.charAt(i)!='@'){
                i++;
            }
            int len = Integer.parseInt(s.substring(start,i));
            res.add(s.substring(i+1,i+1+len));
            i+=len+1;
        }
        return res;
    }
}

class DoubleListNode1{
    public int val;
    public int key;
    public DoubleListNode1 next;
    public DoubleListNode1 prev;
    public DoubleListNode1(int key,int val){
        this.val=val;
        this.key=key;
    }
}
class LRUCache1 {

    DoubleListNode head = null;
    DoubleListNode tail = null;
    int _capacity;
    Map<Integer,DoubleListNode>map=null;
    public LRUCache1(int capacity) {
        map=new HashMap<>();
        _capacity=capacity;
        head=new DoubleListNode(0,0);
        tail=new DoubleListNode(0,0);
        head.next=tail;
        tail.prev=head;
    }

    public int get(int key) {
        if(map.containsKey(key)){
            DoubleListNode node = map.get(key);
            int val = node.val;
            //UNLINK and insert it into head
            insertHead(node);
            return val;
        }else
            return -1;
    }

    public void insertHead(DoubleListNode node){
        node.prev.next=node.next;
        node.next.prev=node.prev;
        node.next=head.next;
        head.next.prev=node;
        node.prev=head;
        head.next=node;
    }

    public void put(int key, int value) {
        if(map.containsKey(key)){
            DoubleListNode node = map.get(key);
            node.val=value;
            insertHead(node);
        }else{
            if(map.size()>=_capacity){
                DoubleListNode deleteNode = tail.prev;
                deleteNode.prev.next=tail;
                tail.prev=deleteNode.prev;
                deleteNode.prev=null;
                deleteNode.next=null;
                map.remove(deleteNode.key);
            }
            DoubleListNode node = new DoubleListNode(key,value);
            node.next=head.next;
            head.next.prev=node;
            node.prev=head;
            head.next=node;
            map.put(key,node);
        }
    }
}


////serialize and deserialize tree
//class Codec {
//
//    // Encodes a tree to a single string.
//    public String serialize(TreeNode root) {
//        StringBuilder sb = new StringBuilder();
//        serialize(root,sb);
//        return sb.toString();
//    }
//    public void serialize(TreeNode root,StringBuilder sb){
//        if(root==null){
//            sb.append('@').append(' ');
//            return;
//        }
//        sb.append(root.val).append(' ');
//        serialize(root.left,sb);
//        serialize(root.right,sb);
//    }
//
//    // Decodes your encoded data to tree.
//    public TreeNode deserialize(String data) {
//        String []datas = data.split(" ");
//        Deque<String>dq = new LinkedList<>(Arrays.asList(datas));
//        return deserialize(dq);
//    }
//
//    public TreeNode deserialize( Deque<String>dq){
//        if(dq.isEmpty())
//            return null;
//        String str = dq.pollFirst();
//        if(str.equals("@")){
//            return null;
//        }
//        TreeNode root = new TreeNode(Integer.parseInt(str));
//        root.left = deserialize(dq);
//        root.right = deserialize(dq);
//        return root;
//    }
//}

class PrintByColumn{
    class Pair{
        int col;
        int val;
        public Pair(int val,int col){
            this.val = val;
            this.col = col;
        }
    }
    private int mostLeft = 0;
    public void print(TreeNode root){
        List<List<Pair>>paths = new ArrayList<>();
        helper(root,0,new ArrayList<>(),paths);
        for(List<Pair>path:paths){
            for(Pair pair:path){
                int gap = -mostLeft-(-pair.col);
                while(gap>0){
                    System.out.print("*");
                    gap--;
                }
                System.out.println(pair.val);
            }
            System.out.println();
        }
    }

    private void helper(TreeNode root, int col, List<Pair>list, List<List<Pair>>paths){
        list.add(new Pair(root.val,col));
        mostLeft = Math.min(mostLeft,col);
        if(root.left == null && root.right==null)
            paths.add(new ArrayList<>(list));
        if(root.left!=null)
            helper(root.left,col-1,list,paths);
        if(root.right!=null)
            helper(root.right,col+1,list,paths);
        list.remove(list.size()-1);
    }
}


class Tuple1{
    public char x;
    public int y;
    public Tuple1(char x,int y){
        this.x = x;
        this.y = y;
    }
}

class MinStackFace {

    private int minVal=0;
    private Stack<Integer>stk = null;
    /** initialize your data structure here. */
    public MinStackFace() {
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





//173 binary search tree iterator
//bst iterator
//class BSTIterator {
//
//
//    //inorder
//    public TreeNode cur=null;
//    public Stack<TreeNode>stk=null;
//    public BSTIterator(TreeNode root) {
//        cur=root;
//        stk=new Stack<>();
//    }
//
//    /** @return whether we have a next smallest number */
//    public boolean hasNext() {
//        return cur!=null || !stk.isEmpty();
//    }
//
//    /** @return the next smallest number */
//    public int next() {
//        while(cur!=null){
//            stk.push(cur);
//            cur=cur.left;
//        }
//        cur=stk.pop();
//        int val=cur.val;
//        cur=cur.right;
//        return val;
//    }
//}

class BinaryPreorderIterator{

    private Stack<TreeNode>stk=null;
    public BinaryPreorderIterator(TreeNode root){
        stk = new Stack<>();
        if(root!=null)
            stk.push(root);
    }

    public boolean hasNext(){
        return !stk.isEmpty();
    }

    public int next(){
        TreeNode node = stk.pop();
        int val =node.val;
        if(node.right!=null)
            stk.push(node.right);
        if(node.left!=null)
            stk.push(node.left);
        return val;
    }
}

class BinaryPostOrderIterator{
    private Stack<TreeNode>stk = null;
    private TreeNode pre =null;
    private TreeNode cur =null;
    public BinaryPostOrderIterator(TreeNode root){
        stk = new Stack<>();
        cur = root;
    }

    public boolean hasNext(){
        return cur!=null||!stk.isEmpty();
    }

    public int next(){
        int val = 0;
        while(true){
            while(cur!=null){
                stk.push(cur);
                cur=cur.left;
            }
            cur = stk.peek();
            if(cur.right!=null && pre!=cur.right){
                cur = cur.right;
            }else{
                val = cur.val;
                stk.pop();
                pre = cur;
                cur =null;
                break;
            }
        }
        return val;
    }
}


//word dictionary


class WordDictionary {

    class TrieNode{
        boolean isEnd;
        TrieNode []children;
        public TrieNode(){
            isEnd=false;
            children=new TrieNode[26];
        }
    }

    TrieNode root =null;

    /** Initialize your data structure here. */
    public WordDictionary() {
        root=new TrieNode();
    }

    /** Adds a word into the data structure. */
    public void addWord(String word) {
        char[]ss=word.toCharArray();
        TrieNode node = root;
        for(char c:ss){
            TrieNode child = node.children[c-'a'];
            if(child==null)
                node.children[c-'a']=new TrieNode();
            node=node.children[c-'a'];
        }
        node.isEnd=true;
    }

    /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
    public boolean search(String word) {
        return find(root,word,0);
    }

    public boolean find(TrieNode node,String word,int ind){
        if(node==null)
            return false;
        if(node!=null && ind==word.length())
            return node.isEnd;
        if(word.charAt(ind)=='.'){
            //traverse all possible children
            for(TrieNode child:node.children){
                if(child!=null && find(child,word,ind+1))
                    return true;
            }
            return false;
        }else{
            if(node.children[word.charAt(ind)-'a']==null)
                return false;
            return find(node.children[word.charAt(ind)-'a'],word,ind+1);
        }

    }
}

// 308 range sum query 2d- mutable

class NumMatrix {

    //two dimensions
    private int[][]sum=null;
    private int[][]numbers=null;
    public int lowbit(int x){
        return x&(-x);
    }

    public void add(int x,int y,int val){
        int m = numbers.length;
        int n = numbers[0].length;
        for(int i=x;i<=m;i+=lowbit(i)){
            for(int j=y;j<=n;j+=lowbit(j)){
                sum[i][j]+=val;
            }
        }
    }

    public int getSum(int x,int y){
        int res =0;
        for(int i=x;i>0;i-=lowbit(i)){
            for(int j=y;j>0;j-=lowbit(j)){
                res+=sum[i][j];
            }
        }
        return res;
    }
    public NumMatrix(int[][] matrix) {
        if(matrix.length==0||matrix[0].length==0)
            return;
        int m = matrix.length,n=matrix[0].length;
        numbers=new int[m][n];
        sum=new int[m+1][n+1];
        for(int i=0;i<m;++i){
            numbers[i]=matrix[i].clone();
            for(int j=0;j<n;++j){
                add(i+1,j+1,matrix[i][j]);
            }
        }
    }

    public void update(int row, int col, int val) {
        add(row+1,col+1,val-numbers[row][col]);
        numbers[row][col]=val;
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        //return getSum(row2+1,col2+1)-getSum(row1,col1);
        return getSum(row2+1,col2+1)+getSum(row1,col1)-getSum(row1,col2+1)-getSum(row2+1,col1);
    }
}




//341 flatten nested list iterator

class NestedIterator implements Iterator<Integer> {

    private Stack<NestedInteger> stk=null;
    public NestedIterator(List<NestedInteger> nestedList) {
        stk=new Stack<>();
        int n=nestedList.size();
        for(int i=n-1;i>=0;--i){
            if(!nestedList.get(i).isInteger()){
                if(nestedList.get(i).getList().isEmpty())
                    continue;
            }
            stk.push(nestedList.get(i));
        }

    }

    @Override
    public Integer next() {
        return stk.pop().getInteger();
    }

    @Override
    public boolean hasNext() {
        while(!stk.isEmpty() && !stk.peek().isInteger()){
            NestedInteger top=stk.pop();

            List<NestedInteger>nestedList=top.getList();
            int n=nestedList.size();
            for(int i=n-1;i>=0;--i){
                if(!nestedList.get(i).isInteger()){
                    if(nestedList.get(i).getList().isEmpty())
                        continue;
                }
                stk.push(nestedList.get(i));
            }

        }
        return !stk.isEmpty();
    }
}

//flatten 2d vector
class Vector2D implements Iterator<Integer> {


    private Queue<Iterator<Integer>>q=null;
    public Vector2D(List<List<Integer>>vec2d){
        q=new LinkedList<>();
        for(List<Integer>vec:vec2d){
            q.offer(vec.iterator());
        }
    }
    public Integer next(){
        return q.peek().next();
    }

    public boolean hasNext(){
        while(!q.isEmpty() && !q.peek().hasNext()){
            q.poll();
        }
        return !q.isEmpty() && q.peek().hasNext();
    }
}


class ZigzagIterator {

    Queue<Iterator>q=null;
    public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
        q=new LinkedList<>();
        q.offer(v1.iterator());
        q.offer(v2.iterator());
    }

    public int next() {
        Iterator<Integer>it = q.poll();
        int val = it.next();
        if(it.hasNext())
            q.offer(it);
        return val;
    }

    public boolean hasNext() {
        while(!q.isEmpty() && !q.peek().hasNext())
            q.poll();
        return !q.isEmpty() && q.peek().hasNext();
    }
}

//348 design tic-tac-toe
class TicTacToe {

    /** Initialize your data structure here. */
    public int[][]rows=null;
    public int[][]cols=null;
    public int[]diagnoal=null;
    public int[]antidiagnoal=null;
    public TicTacToe(int n) {
        rows=new int[n][3];
        cols=new int[n][3];
        diagnoal=new int[3];
        antidiagnoal=new int[3];
    }

    /** Player {player} makes a move at ({row}, {col}).
     @param row The row of the board.
     @param col The column of the board.
     @param player The player, can be either 1 or 2.
     @return The current winning condition, can be either:
     0: No one wins.
     1: Player 1 wins.
     2: Player 2 wins. */
    public int move(int row, int col, int player) {
        rows[row][player]++;
        if(rows[row][player]==rows.length)
            return player;
        cols[col][player]++;
        if(cols[col][player]==cols.length)
            return player;
        if(row==col)
            diagnoal[player]++;
        if(diagnoal[player]==rows.length)
            return player;
        if(row+col==rows.length-1)
            antidiagnoal[player]++;
        if(antidiagnoal[player]==rows.length)
            return player;
        return 0;

    }
}


//380 insert delete getRandom O(1)
class RandomizedSet {

    public List<Integer>num=null;
    Random random =null;
    public Map<Integer,Integer>map=null;
    /** Initialize your data structure here. */
    public RandomizedSet() {
        num=new ArrayList<>();
        map=new HashMap<>();
        random=new Random();
    }

    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if(map.containsKey(val))
            return false;
        map.put(val,num.size());
        num.add(val);
        return true;
    }

    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        if(!map.containsKey(val))
            return false;
        //change with last index
        int size = num.size();
        if(map.get(val)!=size-1){
            int tail = num.get(size-1);
            int ind = map.get(val);
            map.put(tail,ind);
            num.set(ind,tail);
        }
        map.remove(val);
        num.remove(size-1);
        return true;
    }

    /** Get a random element from the set. */
    public int getRandom() {
        int size = num.size();
        return num.get(random.nextInt(size));
    }
}


//381 insert delete getrandom o(1)- duplicates allowed
class RandomizedCollection {
    ArrayList<Integer> nums;
    HashMap<Integer, Set<Integer>> locs;
    java.util.Random rand = new java.util.Random();

    /**
     * Initialize your data structure here.
     */
    public RandomizedCollection() {
        nums = new ArrayList<Integer>();
        locs = new HashMap<Integer, Set<Integer>>();
    }

    /**
     * Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
     */
    public boolean insert(int val) {
        boolean contain = locs.containsKey(val);
        if (!contain) locs.put(val, new LinkedHashSet<Integer>());
        locs.get(val).add(nums.size());
        nums.add(val);
        return !contain;
    }

    /**
     * Removes a value from the collection. Returns true if the collection contained the specified element.
     */
    public boolean remove(int val) {
        boolean contain = locs.containsKey(val);
        if (!contain) return false;
        int loc = locs.get(val).iterator().next();
        locs.get(val).remove(loc);
        if (loc < nums.size() - 1) {
            int lastone = nums.get(nums.size() - 1);
            nums.set(loc, lastone);
            locs.get(lastone).remove(nums.size() - 1);
            locs.get(lastone).add(loc);
        }
        nums.remove(nums.size() - 1);

        if (locs.get(val).isEmpty()) locs.remove(val);
        return true;
    }

    /**
     * Get a random element from the collection.
     */
    public int getRandom() {
        return nums.get(rand.nextInt(nums.size()));
    }
}

public class Facebook {

    //301 remove invalid parenthesis
    //bfs 传统的方法

    public boolean isValid(String s){
        int cnt=0,n=s.length();
        for(int i=0;i<n;++i){
            if(s.charAt(i)!='(' && s.charAt(i)!=')')
                continue;
            if(s.charAt(i)=='(')
                cnt++;
            else
                cnt--;
            if(cnt<0)
                return false;
        }
        return cnt==0;
    }
    public List<String> removeInvalidParentheses(String s) {
        //bfs way
        //dfs way
        Queue<String> q=new LinkedList<>();
        Set<String> vis=new HashSet<>();
        q.offer(s);
        vis.add(s);
        List<String>res=new ArrayList<>();
        boolean hasNext=true;
        while(!q.isEmpty() && hasNext){
            int size = q.size();
            for(int i=0;i<size;++i){
                String top = q.poll();
                if(isValid(top)){
                    hasNext=false;
                    res.add(top);
                }else{
                    int n=top.length();
                    for(int ii=0;ii<n;++ii){
                        if(top.charAt(ii)=='('||top.charAt(ii)==')'){
                            String substr = top.substring(0,ii)+top.substring(ii+1);
                            if(!vis.contains(substr)){
                                vis.add(substr);
                                q.offer(substr);
                            }
                        }
                    }
                }
            }
        }
        return res;
    }

    //dfs way
    public List<String> removeInvalidParenthesesDFS(String s) {
        List<String> ans = new ArrayList<>();
        remove(s, ans, 0, 0, new char[]{'(', ')'});
        return ans;
    }

    public void remove(String s, List<String> ans, int last_i, int last_j,  char[] par) {
        for (int stack = 0, i = last_i; i < s.length(); ++i) {
            if (s.charAt(i) == par[0]) stack++;
            if (s.charAt(i) == par[1]) stack--;
            if (stack >= 0) continue;
            for (int j = last_j; j <= i; ++j)
                if (s.charAt(j) == par[1] && (j == last_j || s.charAt(j - 1) != par[1]))
                    remove(s.substring(0, j) + s.substring(j + 1, s.length()), ans, i, j, par);
            return;
        }
        String reversed = new StringBuilder(s).reverse().toString();
        if (par[0] == '(') // finished left to right
            remove(reversed, ans, 0, 0, new char[]{')', '('});
        else // finished right to left
            ans.add(reversed);
    }


    //just return one valid
    //stack way
    public int minimumDeleteTimes(String s){
        int count =0;
        char []ss = s.toCharArray();
        for(char c:ss){
            if(c=='(')
                count++;
            else if(c==')')
                count--;
        }
        return count>0?count:-count;
    }


    public String turnToValid(String s){
        char []ss = s.toCharArray();
        Stack<Integer>stk = new Stack<>();
        HashSet<Integer>set = new HashSet<>();//index need to delete;
        int n = ss.length;
        for(int i=0;i<n;++i){
            if(ss[i]=='(')
                stk.push(i);
            else if(ss[i]==')'){
                if(stk.isEmpty())
                    set.add(i);
                else
                    stk.pop();
            }
        }

        while(!stk.isEmpty()){
            set.add(stk.pop());
        }

        StringBuilder sb = new StringBuilder();
        for(int i=0;i<n;++i){
            if(!set.contains(i))
                sb.append(ss[i]);
        }
        return sb.toString();
    }


    //without stk
    public String deleteCloseParenthese(String input){
        int count = 0;
        StringBuilder sb = new StringBuilder();
        char []ss = input.toCharArray();
        for(char c:ss){
            sb.append(c);
            if(c=='(')
                count++;
            else if(c==')'){
                if(count>0)
                    count--;
                else
                    sb.deleteCharAt(sb.length()-1);
            }
        }
        return sb.toString();
    }


    public String deleteOpenParenthese(String input){
        int count =0 , n = input.length();
        StringBuilder sb = new StringBuilder();
        for(int i=n-1;i>=0;--i){
            sb.append(input.charAt(i));
            if(input.charAt(i)==')')
                count++;
            else if(input.charAt(i)=='('){
                if(count>0)
                    count--;
                else
                    sb.deleteCharAt(sb.length()-1);
            }
        }
        sb.reverse();
        return sb.toString();
    }

    public String balanceParenthese(String input){
        String ans = deleteCloseParenthese(input);
        ans = deleteOpenParenthese(ans);
        return ans;
    }


    //3sum without sort

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
                    res.add(Arrays.asList(new Integer[]{nums[i],nums[begin],nums[end]}));
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

    //without sort
    public List<List<Integer>> threeSum(int[]nums,int target){
        List<List<Integer>>ans = new ArrayList<>();
        Set<List<Integer>>visited = new HashSet<>();
        Map<Integer,Integer> valueToIndex = new HashMap<>();
        int n = nums.length;
        for(int i=0;i<n;++i)
            valueToIndex.put(nums[i],i);
        for(int i=0;i<n-1;++i){
            int newTarget = target-nums[i];
            for(int j=i+1;j<n;++j){
                if(valueToIndex.containsKey(newTarget-nums[j])){
                    int index = valueToIndex.get(newTarget-nums[j]);
                    if(index!=j && index!=i){
                        List<Integer>pair = new ArrayList<>(Arrays.asList(nums[i],nums[j],nums[index]));
                        Collections.sort(pair);
                        if(!visited.contains(pair)){
                            ans.add(pair);
                            visited.add(pair);
                        }
                    }
                }
            }
        }
        return ans;
    }


    //letter combinations of a phone number
    public void dfs(List<String>res,String path,String digits,int ind,String[]args){
        if(ind==digits.length()){
            res.add(path);
            return;
        }
        for(int i=0;i<args[digits.charAt(ind)-'2'].length();++i){
            dfs(res,path+args[digits.charAt(ind)-'2'].charAt(i),digits,ind+1,args);
        }
    }
    public List<String> letterCombinations(String digits) {
        List<String>res = new ArrayList<>();
        if(digits.isEmpty())
            return res;
        String []args={"abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
        dfs(res,"",digits,0,args);
        return res;
    }



    //67 add binary
    public String addBinary(String a, String b) {
        //simple way
        char[]aa=a.toCharArray();
        char[]bb=b.toCharArray();
        int m =aa.length;
        int n = bb.length;
        StringBuilder sb = new StringBuilder();
        int i=m-1,j=n-1;
        int carry=0;
        while(i>=0||j>=0||carry>0){
            int sum = (i>=0?aa[i--]-'0':0)+(j>=0?bb[j--]-'0':0)+carry;
            carry=sum/2;
            sb.append(sum%2);
        }
        return sb.reverse().toString();
    }

    //add binary not using +-*/. just bit manipulation
    public String addBinaryWithBit(String a, String b){
        StringBuilder sb = new StringBuilder();
        int carry = 0, i = a.length()-1, j =b.length()-1;
        while(i>=0||j>=0||carry>0){
            int num1 = i>=0?a.charAt(i--)-'0':0;
            int num2 = j>=0?b.charAt(j--)-'0':0;
            int sum = carry^num1^num2;
            carry = (carry&num1)|(num1&num2)|(num2&carry);
        }
        sb.reverse();
        return sb.toString();
    }

    //311 sparse matrix multiplication
    public int[][] multiply(int[][] A, int[][] B) {
        //A:m*n , B:n*k;
        int m = A.length,n=A[0].length,k=B[0].length;
        int[][]res=new int[m][k];
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(A[i][j]==0)
                    continue;
                for(int kk=0;kk<k;++kk){
                    res[i][kk]+=A[i][j]*B[j][kk];
                }
            }
        }
        return res;
    }

    //43 multiple strings
    public String multiplyStrings(String num1, String num2) {
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


    //621 task scheduler 好好研究一次，顺序不能变

    //two sum

    //76 minimum window substring
    public String minWindow(String s, String t) {
        int m = s.length(),n=t.length();
        int start=0,len=Integer.MAX_VALUE;
        int[]cnt=new int[128];
        char[]ss=s.toCharArray();
        char[]tt=t.toCharArray();
        for(char c:tt){
            cnt[c]++;
        }
        int begin=0,end=0;
        while(end<m){
            if(cnt[ss[end++]]-- >0)
                n--;
            while(n==0){
                if(len>end-begin){
                    start=begin;
                    len=end-begin;
                }
                if(++cnt[ss[begin++]]>0)
                    n++;
            }
        }
        return len==Integer.MAX_VALUE?"":s.substring(start,start+len);
    }


    //125 valid palindrome
    public boolean isPalindrome(String s) {
        char[]ss=s.toCharArray();
        int n = ss.length;
        int left=0,right=n-1;
        while(left<right){
            while(left<right && !Character.isLetterOrDigit(ss[left]))
                left++;
            while(left<right && !Character.isLetterOrDigit(ss[right]))
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


    //253 meeting rooms II
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

    //priority queue way



    //273 integer to english word
    //english to words
    //number to words
    public String toWords(int num,String[]args,String[]tens){
        int ind=0;
        if(num>=1000000000){
            ind = num/1000000000;
            return toWords(ind,args,tens)+" Billion"+toWords(num-ind*1000000000,args,tens);
        }else if(num>=1000000){
            ind = num/1000000;
            return toWords(ind,args,tens)+" Million"+toWords(num-ind*1000000,args,tens);
        }else if(num>=1000){
            ind = num/1000;
            return toWords(ind,args,tens)+" Thousand"+toWords(num-ind*1000,args,tens);
        }else if(num>=100){
            ind = num/100;
            return toWords(ind,args,tens)+" Hundred"+ toWords(num-ind*100,args,tens);
        }else if(num>=20){
            return " "+tens[num/10-2]+toWords(num%10,args,tens);
        }else if(num>0){
            return " "+args[num];
        }else{
            return "";
        }

    }

    public String numberToWords(int num) {
        String[]args={"Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Eleven","Twelve","Thirteen","Fourteen","Fifteen","Sixteen","Seventeen","Eighteen","Nineteen"};
        String[]tens={"Twenty","Thirty","Forty","Fifty","Sixty","Seventy","Eighty","Ninety"};
        if(num==0)
            return "Zero";
        String res=toWords(num,args,tens);
        return res.substring(1);
    }



    //314
    class Tuple{
        public int level;
        public TreeNode node;
        public Tuple(TreeNode node,int level){
            this.level=level;
            this.node=node;
        }
    }
    public void getRange(TreeNode root,int level,int[]res){
        if(root==null)
            return;
        res[0]=Math.min(res[0],level);
        res[1]=Math.max(res[1],level);
        getRange(root.left,level-1,res);
        getRange(root.right,level+1,res);
    }
    public List<List<Integer>> verticalOrder(TreeNode root) {
        //bfs
        List<List<Integer>>res=new ArrayList<>();
        if(root==null)
            return res;
        int[]range={0,0};
        getRange(root,0,range);
        for(int i=range[0];i<=range[1];++i)
            res.add(new ArrayList<>());
        Queue<Tuple>q=new LinkedList<>();
        q.offer(new Tuple(root,0));
        while(!q.isEmpty()){
            Tuple top = q.poll();
            int index = top.level-range[0];
            res.get(index).add(top.node.val);
            if(top.node.left!=null)
                q.offer(new Tuple(top.node.left,top.level-1));
            if(top.node.right!=null)
                q.offer(new Tuple(top.node.right,top.level+1));
        }
        return res;
    }




    //sort colors
    public void swap(int[]nums,int x,int y){
        int tmp = nums[x];
        nums[x]=nums[y];
        nums[y]=tmp;
    }
    public void sortColors(int[] nums) {
        int n= nums.length;
        int left=0,right=n-1,ind=0;
        while(ind<=right){
            if(nums[ind]==0){
                swap(nums,ind,left);
                ind++;
                left++;
            }else if(nums[ind]==1)
                ind++;
            else{
                swap(nums,ind,right);
                right--;
            }
        }
    }

    //sort k color

    public void swap(List<Integer>colors,int i,int j){
        int x = colors.get(i);
        colors.set(i,colors.get(j));
        colors.set(j,x);
    }
    public void sortKcolor(List<Integer>colors,int k){
        //每次干掉俩
        int left = 0;
        int right = colors.size()-1;
        while(k>0){
            int min1 = Integer.MAX_VALUE;
            int max1 = Integer.MIN_VALUE;
            for(int i=left; i<=right;++i){
                min1 = Math.min(min1,colors.get(i));
                max1 = Math.max(max1,colors.get(i));
            }
            int index = left;
            int minIndex = left;
            int maxIndex  = right;
            while(minIndex<maxIndex && index<=maxIndex){
                if(colors.get(index)==min1){
                    swap(colors,minIndex,index);
                    minIndex++;
                    index++;
                }else if(colors.get(index)==max1){
                    swap(colors,index,maxIndex);
                    maxIndex--;
                }else
                    index++;
            }
            left = minIndex;
            right = maxIndex;
            k-=2;
        }
    }

    //count sort
    public void sortColors2(int[]colors,int k){
        int []count = new int[k];
        for(int color:colors){
                count[color-1]++;
        }
        int index =0;
        for(int i=0;i<k;++i){
            while(count[i]>0){
                colors[index++]=i+1;
                count[i]--;
            }
        }
    }



    //follow up， int getCategory(int n，outputs the category 1 to k of given n

    public int getCategory(int n){
        return 1;
    }
    public void sortKcolors(int[]nums,int k){
        if(nums==null||nums.length<=1||k<=1)
            return;
        int left = 0;
        int right = nums.length-1;
        int min =1;
        int max=k;
        while(left<right){
            int i = left;
            while(i<=right){
                if(getCategory(nums[i])==min){
                    swap(nums,i,left);
                    i++;
                    left++;
                }else if(getCategory(nums[i])>min && getCategory(nums[i])<max)
                    i++;
                else{
                    swap(nums,i,right);
                    right--;
                }
            }
            min++;
            max--;
        }

    }




    //98 validate bst
    TreeNode pre =null;
    public boolean isValidBST(TreeNode root) {
        if(root==null)
            return true;
        if(!isValidBST(root.left))
            return false;
        if(pre!=null && pre.val>=root.val)
            return false;
        pre = root;
        return isValidBST(root.right);
    }

    public boolean isValidBSTBFS(TreeNode root) {
        if (root == null) return true;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode pre = null;
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            if(pre != null && root.val <= pre.val) return false;
            pre = root;
            root = root.right;
        }
        return true;
    }

    //clone graph
    Map<UndirectedGraphNode,UndirectedGraphNode>UndirectedMap= new HashMap<>();
    public UndirectedGraphNode cloneGraphDFS(UndirectedGraphNode node) {
        if(node==null)
            return node;
        if(!UndirectedMap.containsKey(node)){
            UndirectedMap.put(node,new UndirectedGraphNode(node.label));
            for(UndirectedGraphNode neighbor:node.neighbors){
                UndirectedMap.get(node).neighbors.add(cloneGraphDFS(neighbor));
            }
        }
        return UndirectedMap.get(node);
    }

    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        //map的就不说了
        //bfs
        Map<UndirectedGraphNode,UndirectedGraphNode>map = new HashMap<>();
        if(node==null)
            return null;
        Queue<UndirectedGraphNode>q = new LinkedList<>();
        q.offer(node);
        while(!q.isEmpty()){
            UndirectedGraphNode cur = q.poll();
            if(!map.containsKey(cur))
                map.put(cur,new UndirectedGraphNode(cur.label));
            for(UndirectedGraphNode ne:cur.neighbors){
                if(!map.containsKey(ne)){
                    q.offer(ne);
                    map.put(ne,new UndirectedGraphNode(ne.label));
                }
                map.get(cur).neighbors.add(map.get(ne));
            }
        }
        return map.get(node);
    }

    //139 word break
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String>set = new HashSet<>();
        set.addAll(wordDict);
        int n = s.length();
        boolean []dp = new boolean [n+1];
        dp[0]=true;
        for(int i=1;i<=n;++i){
            for(int j=i-1;j>=0;--j){
                if(set.contains(s.substring(j,i)) && dp[j]){
                    dp[i]=true;
                    break;
                }
            }
        }
        // for(int i=0;i<=n;++i)
        //     System.out.println(dp[i]);
        return dp[n];

    }

    //number of islands
    public void dfs(char[][]grid,int x,int y,boolean[][]vis){
        if(x<0||y<0||x>=grid.length||y>=grid[0].length||vis[x][y]||grid[x][y]!='1')
            return;
        vis[x][y]=true;
        dfs(grid,x+1,y,vis);
        dfs(grid,x-1,y,vis);
        dfs(grid,x,y+1,vis);
        dfs(grid,x,y-1,vis);
    }
    public int numIslands(char[][] grid) {
        ///dfs
        if(grid.length==0||grid[0].length==0)
            return 0;
        int m = grid.length,n=grid[0].length;
        boolean [][]vis=new boolean[m][n];
        int cnt=0;
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(grid[i][j]=='1' && !vis[i][j]){
                    cnt++;
                    dfs(grid,i,j,vis);
                }
            }
        }
        return cnt;
    }

    //bfs way
    public int numIslandsBFS(char[][] grid) {
        if(grid.length==0||grid[0].length==0)
            return 0;
        int m = grid.length,n=grid[0].length;
        int []dx={1,-1,0,0};
        int []dy ={0,0,1,-1};
        boolean [][]vis=new boolean[m][n];
        Queue<int[]>q=new LinkedList<>();
        int cnt=0;
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                if(!vis[i][j] && grid[i][j]=='1'){
                    vis[i][j]=true;
                    cnt++;
                    q.offer(new int[]{i,j});
                    while(!q.isEmpty()){
                        int []top=q.poll();
                        int x =top[0];
                        int y=top[1];
                        for(int k=0;k<4;++k){
                            int nx=x+dx[k];
                            int ny=y+dy[k];
                            if(nx<0||nx>=m||ny<0||ny>=n||vis[nx][ny]||grid[nx][ny]!='1')
                                continue;
                            vis[nx][ny]=true;
                            q.offer(new int[]{nx,ny});
                        }
                    }
                }
            }
        }
        return cnt;
    }

    //277 find the celebrity

    public boolean knows(int x,int y){
        return true;
    }
    public int findCelebrity(int n) {
        int candidate=0;
        for(int i=1;i<n;++i){
            if(knows(candidate,i))
                candidate=i;
        }
        //to check
        for(int i=0;i<n;++i){
            if(i==candidate)
                continue;
            if(knows(candidate,i)||!knows(i,candidate))
                return -1;
        }
        return candidate;
    }


    //278 first bad version
    public boolean isBadVersion(int x){
        return true;
    }
    public int firstBadVersion(int n) {
        int begin=0,end=n;
        while(begin<end){
            int mid=(end-begin)/2+begin;
            if(isBadVersion(mid)){
                end=mid;
            }else
                begin=mid+1;
        }
        return begin;
    }

    //282 expression add operators
    public void backtrack(List<String>res,String num,int target,long cumulative,long cur,int pos,String path){
        if(pos==num.length() && target==cumulative){
            res.add(path);
            return;
        }
        for(int i=pos+1;i<=num.length();++i){
            if (i != pos+1 && num.charAt(pos) == '0')
                break;//continue;//去掉lead zero
            String sub=num.substring(pos,i);
            long val=Long.parseLong(sub);
            if(pos==0)
                backtrack(res,num,target,val,val,i,sub);
            else{
                //three cases
                backtrack(res,num,target,cumulative+val,val,i,path+"+"+sub);
                backtrack(res,num,target,cumulative-val,-val,i,path+"-"+sub);
                backtrack(res,num,target,cumulative-cur+cur*val,cur*val,i,path+"*"+sub);
            }
        }
    }
    public List<String> addOperators(String num, int target) {
        List<String>res=new ArrayList<>();
        backtrack(res,num,target,0,0,0,"");
        return res;
    }


    //bst to sorted circular double linkedlist
    //attention

    //stack way
    public TreeNode bst2DoubleList(TreeNode root){
        TreeNode dummy = new TreeNode(0);
        TreeNode prev = dummy;
        Stack<TreeNode>stk = new Stack<>();
        while(root!=null || !stk.isEmpty()){
            while(root!=null){
                stk.push(root);
                root = root.left;
            }
            root = stk.pop();
            prev.right = root;
            root.left = prev;
            prev=root;
            root=root.right;
        }

        //if circular , add:
        prev.right = dummy.right;
        dummy.right.left = prev;
        return dummy.right;
    }

    ///dfs way;

    private TreeNode prev=null;
    public void inorder(TreeNode node){
        if(node==null)
            return;
        inorder(node.left);
        node.left = prev;
        prev.right = node;
        prev = node;
        inorder(node.right);
    }
    public TreeNode bst2DoubleListRecursive(TreeNode node){
        TreeNode dummy = new TreeNode(0);
        prev = dummy;
        inorder(node);
        prev.right = dummy.right;
        dummy.right.left = prev;
        return dummy.right;
    }

    //29 divide two integers
    public int divide(int dividend, int divisor) {
        long divid = (long)dividend;
        long divis = (long)divisor;
        boolean isNegative = (divid>=0)^(divis>=0);
        if(dividend==0)
            return 0;
        divid =Math.abs(divid);
        divis = Math.abs(divis);
        long sum=0;
        while(divid>=divis){
            int ind=0;
            long tmp = divis;
            while(divid>=tmp){
                divid-=tmp;
                sum+=(long)(1<<ind);
                ind++;
                tmp<<=1;
            }
        }
        sum=isNegative?-sum:sum;
        if(sum>Integer.MAX_VALUE)
            return Integer.MAX_VALUE;
        return (int)sum;
    }

    //33 search in rotated sorted array
    public int search(int[] nums, int target) {
        int n = nums.length;
        if(n==0)
            return -1;
        int begin=0, end = n-1;
        while(begin<end){
            int mid = (end-begin)/2+begin;
            if(nums[mid]==target)
                return mid;
            if(nums[mid]>nums[end]){
                if(nums[mid]>target && target>=nums[begin])
                    end=  mid;
                else
                    begin = mid+1;
            }else{
                if(nums[mid]<target && target<=nums[end])
                    begin = mid+1;
                else
                    end=  mid;
            }
        }
        return nums[begin]==target?begin:-1;
    }


    //permutation II
    //try to find a better way
    public void dfs(List<List<Integer>>res,int[]nums,List<Integer>path){
        if(path.size()==nums.length){
            res.add(new ArrayList<>(path));
            return;
        }
        for(int i=0;i<nums.length;++i){
            if(i>0 && nums[i]==nums[i-1])
                continue;
            int c1=0,c2=0;
            for(int x:nums){
                if(x==nums[i])
                    c1++;
            }
            for(int x:path){
                if(x==nums[i])
                    c2++;
            }
            if(c1>c2){
                path.add(nums[i]);
                dfs(res,nums,path);
                path.remove(path.size()-1);
            }
        }
    }
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>>res=new ArrayList<>();
        Arrays.sort(nums);
        dfs(res,nums,new ArrayList<>());
        return res;
    }


    //merge intervals
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

    //efficient way
    public List<Interval> mergeEfficient(List<Interval> intervals) {
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

    //calculate the time
    public int totalTime(List<Interval>intervals){
        if(intervals==null||intervals.size()==0)
            return 0;
        Collections.sort(intervals, new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return o1.start-o2.start;
            }
        });
        int total = 0;
        Interval prev = new Interval(0,0);
        for(Interval cur:intervals){
            if(prev.end<=cur.start){
                total+=cur.end-cur.start;
                prev= cur;
            }else if(cur.end>prev.end){
                total += cur.end-prev.end;
                prev=cur;
            }
        }
        return total;
    }

    //valid number
    public boolean isNumber(String s) {
        char[]ss=s.toCharArray();
        int end = ss.length;
        int n=ss.length;
        int begin=0;
        //delete space
        while(begin<n && Character.isSpace(ss[begin])){
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
            while(begin<end && Character.isDigit(ss[begin])){
                num++;
                begin++;
            }
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

        while(begin<end && Character.isSpace(ss[begin]))
            begin++;
        if(num<1||begin!=end)
            return false;
        if(hasE && index<1)
            return false;
        return true;
    }

    //188 best time to buy and sell stock IV
    //DP: t(i,j) is the max profit for up to i transactions by time j (0<=i<=K, 0<=j<=T).
    public int maxProfit(int k, int[] prices) {
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


    private int quickSolve(int[] prices) {
        int len = prices.length, profit = 0;
        for (int i = 1; i < len; i++)
            // as long as there is a price gap, we gain a profit.
            if (prices[i] > prices[i - 1]) profit += prices[i] - prices[i - 1];
        return profit;
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

    public void rotateSimple(int[] nums, int k) {
        int[]clone = nums.clone();
        int n = nums.length;
        k=k%n;
        for(int i=0;i<n;++i){
            nums[i]=clone[(i+n-k)%n];
        }
    }


    public void rotateBySwap(int[] nums, int k) {
        //two way
        int n = nums.length;
        if(n==0||k%n==0)
            return;
        k=n-(k%n)-1;
        swap(nums,0,k);
        swap(nums,k+1,n-1);
        swap(nums,0,n-1);

    }


    //207 course scheuld
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
    public boolean canFinish(int n, int[][] prerequisites) {
        Map<Integer,List<Integer>>map=new HashMap<>();
        for(int []pre:prerequisites){
            if(!map.containsKey(pre[1]))
                map.put(pre[1],new ArrayList<>());
            map.get(pre[1]).add(pre[0]);
        }
        boolean[]vis=new boolean[n];
        for(int i=0;i<n;++i){
            if(hasCycle(map,i,vis,new boolean[n]))
                return false;
        }
        return true;
    }


    //210 course schedule II
    public boolean hasCycle(Map<Integer,List<Integer>>map,int start,boolean[]vis,boolean[]onLoop,Stack<Integer>stk){
        if(vis[start])
            return false;
        if(onLoop[start])
            return true;
        onLoop[start]=true;
        List<Integer>edges=map.getOrDefault(start,new ArrayList<>());
        for(int edge:edges){
            if(hasCycle(map,edge,vis,onLoop,stk))
                return true;
        }
        vis[start]=true;
        stk.push(start);
        return false;
    }
    public int[] findOrder(int n, int[][] prerequisites) {
        Map<Integer,List<Integer>>map=new HashMap<>();
        for(int[]pre:prerequisites){
            if(!map.containsKey(pre[1])){
                map.put(pre[1],new ArrayList<>());
            }
            map.get(pre[1]).add(pre[0]);
        }

        boolean[]vis=new boolean[n];
        Stack<Integer>stk=new Stack<>();
        for(int i=0;i<n;++i){
            if(hasCycle(map,i,vis,new boolean[n],stk))
                return new int[]{};
        }
        int[]res=new int[n];
        int ind=0;
        while(!stk.isEmpty()){
            res[ind++]=stk.pop();
        }
        return res;
    }


    //bfs course schedule
    public int[] findOrderBFS(int n, int[][] prerequisites) {
        //bfs
        Map<Integer,List<Integer>>map=new HashMap<>();
        int[]indegree=new int[n];
        for(int[]pre:prerequisites){
            if(!map.containsKey(pre[1]))
                map.put(pre[1],new ArrayList<>());
            map.get(pre[1]).add(pre[0]);
            indegree[pre[0]]++;
        }
        Queue<Integer>q=new LinkedList<>();
        for(int i=0;i<n;++i)
            if(indegree[i]==0)
                q.offer(i);
        int cnt=0;
        int []res=new int[n];
        while(!q.isEmpty()){
            int top=q.poll();
            res[cnt++]=top;
            List<Integer>edges=map.getOrDefault(top,new ArrayList<>());
            for(int edge:edges){
                --indegree[edge];
                if(indegree[edge]==0)
                    q.offer(edge);
            }
        }
        return cnt==n?res:new int[]{};
    }


    //269 alien dictionary
    public String alienOrder(String[] words) {
        Map<Character, Set<Character>> map=new HashMap<Character, Set<Character>>();
        Map<Character, Integer> degree=new HashMap<Character, Integer>();
        String result="";
        if(words==null || words.length==0) return result;
        for(String s: words){
            for(char c: s.toCharArray()){
                degree.put(c,0);
            }
        }
        for(int i=0; i<words.length-1; i++){
            String cur=words[i];
            String next=words[i+1];
            int length=Math.min(cur.length(), next.length());
            for(int j=0; j<length; j++){
                char c1=cur.charAt(j);
                char c2=next.charAt(j);
                if(c1!=c2){
                    Set<Character> set=new HashSet<Character>();
                    if(map.containsKey(c1)) set=map.get(c1);
                    if(!set.contains(c2)){
                        set.add(c2);
                        map.put(c1, set);
                        degree.put(c2, degree.get(c2)+1);
                    }
                    break;
                }
            }
        }
        Queue<Character> q=new LinkedList<Character>();
        for(char c: degree.keySet()){
            if(degree.get(c)==0) q.add(c);
        }
        while(!q.isEmpty()){
            char c=q.remove();
            result+=c;
            if(map.containsKey(c)){
                for(char c2: map.get(c)){
                    degree.put(c2,degree.get(c2)-1);
                    if(degree.get(c2)==0) q.add(c2);
                }
            }
        }
        if(result.length()!=degree.size()) return "";
        return result;
    }

    //intersection of two arrays
    //set || sort two pointers || sort binary search




    // 3 longest substring without repeating characters
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

    //hashmap solution
    public int lengthOfLongestSubstringMap(String s) {
        Map<Character,Integer>map=new HashMap<>();
        int start=0,n=s.length(),maxVal=0;
        for(int i=0;i<n;++i){
            if(map.containsKey(s.charAt(i)))
                start=Math.max(start,1+map.get(s.charAt(i)));
            maxVal=Math.max(maxVal,i-start+1);
            map.put(s.charAt(i),i);
        }
        return maxVal;
    }

    //10 regular expression matching
    public boolean isMatch(String s, String p) {
        //recursive way
        if(p.isEmpty())
            return s.isEmpty();
        if(p.length()==1)
            return s.length()==1 && (s.charAt(0)==p.charAt(0)||p.charAt(0)=='.');
        if(s.isEmpty())
            return p.charAt(1)=='*' && isMatch(s,p.substring(2));
        if(p.charAt(1)=='*'){
            return isMatch(s,p.substring(2))||((s.charAt(0)==p.charAt(0)||p.charAt(0)=='.')&&isMatch(s.substring(1),p));
        }else{
            return (s.charAt(0)==p.charAt(0)||p.charAt(0)=='.')&&isMatch(s.substring(1),p.substring(1));
        }
    }

    //dp way, you should save more space, try to do that
    public boolean isMatchDP(String s, String p) {
        //dp
        int m = s.length(),n=p.length();
        boolean [][]dp=new boolean[m+1][n+1];
        dp[0][0]=true;
        for(int i=0;i<=m;++i){
            for(int j=0;j<=n;++j){
                if(j>=2 && p.charAt(j-1)=='*'){
                    dp[i][j]=dp[i][j-2]||(i>=1 &&( p.charAt(j-2)==s.charAt(i-1)||p.charAt(j-2)=='.')&&dp[i-1][j]);
                }else{
                    if(i>=1 && j>=1)
                        dp[i][j]=dp[i-1][j-1] && (p.charAt(j-1)==s.charAt(i-1)||p.charAt(j-1)=='.');
                }
            }
        }
        return dp[m][n];
    }


    // 28 implement strstr()
    public int strStr(String haystack, String needle) {
        char []hh= haystack.toCharArray();
        char []need = needle.toCharArray();
        int n = hh.length,m=need.length;
        if(m==0)
            return 0;
        int i=0;
        for(;i<=n-m;++i){
            if(hh[i]==need[0]){
                int j=0;
                for(;j<m;++j){
                    if(hh[i+j]!=need[j])
                        break;
                }
                if(j==m)
                    return i;
            }
        }
        return -1;
    }

    //39 combination sum
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


    //46
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        // Arrays.sort(nums); // not necessary
        backtrack(list, new ArrayList<>(), nums);
        return list;
    }

    private void backtrack(List<List<Integer>> list, List<Integer> tempList, int [] nums){
        if(tempList.size() == nums.length){
            list.add(new ArrayList<>(tempList));
        } else{
            for(int i = 0; i < nums.length; i++){
                if(tempList.contains(nums[i])) continue; // element already exists, skip
                tempList.add(nums[i]);
                backtrack(list, tempList, nums);
                tempList.remove(tempList.size() - 1);
            }
        }
    }

    //38 count and say: seed would be anything
    public String countAndSay(int n) {
        String s = "1";
        int ind=1;
        while(ind<n){
            char []ss=s.toCharArray();
            StringBuilder sb = new StringBuilder();
            int m = ss.length;
            int cnt=1;
            int i=1;
            while(i<m){
                if(ss[i]!=ss[i-1]){
                    sb.append(cnt).append(ss[i-1]);
                    cnt=1;
                }else
                    cnt++;
                i++;
            }
            if(cnt!=0){
                sb.append(cnt).append(ss[m-1]);
            }
            s=sb.toString();
            ind++;
        }
        return s;
    }

    //51 n-queens
    public boolean check(int[]board,int k){
        for(int i=0;i<k;++i){
            if(board[i]==board[k]||Math.abs(board[i]-board[k])==k-i)
                return false;
        }
        return true;
    }
    public void dfs(List<List<String>>res,int[]board,int ind){
        if(ind==board.length){
            //print point
            List<String>tmp=new ArrayList<>();
            for(int i=0;i<board.length;++i){
                StringBuilder sb =new StringBuilder("");
                for(int j=0;j<board.length;++j)
                    sb.append(j==board[i]?'Q':'.');
                tmp.add(sb.toString());
            }
            res.add(tmp);
            return;
        }
        for(int i=0;i<board.length;++i){
            board[ind]=i;
            if(check(board,ind))
                dfs(res,board,ind+1);
            board[ind]=-1;
        }
    }
    public List<List<String>> solveNQueens(int n) {
        List<List<String>>res=new ArrayList<>();
        int []board=new int[n];
        Arrays.fill(board,-1);
        dfs(res,board,0);
        return res;
    }


    //73 set matrix zeroes
    public void setZeroes(int[][] matrix) {
        if(matrix.length==0||matrix[0].length==0)
            return;
        boolean firstCol =  false;
        int m = matrix.length,n = matrix[0].length;
        for(int i=0;i<m;++i){
            if(matrix[i][0]==0)
                firstCol=true;
            for(int j=1;j<n;++j){
                if(matrix[i][j]==0){
                    matrix[i][0]=0;
                    matrix[0][j]=0;
                }
            }
        }

        for(int i=m-1;i>=0;--i){
            for(int j=n-1;j>=1;--j){
                if(matrix[0][j]==0||matrix[i][0]==0){
                    matrix[i][j]=0;
                }
            }
            if(firstCol)
                matrix[i][0]=0;
        }
    }

    //78 subsets
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>>res=new ArrayList<>();
        int n = nums.length;
        int num = 1<<n;
        for(int i=0;i<num;++i){
            res.add(new ArrayList<>());
            for(int j=0;j<n;++j){
                if(((i>>j)&0x1)!=0)
                    res.get(i).add(nums[j]);
            }
        }
        return res;
    }

    public void dfs78(List<List<Integer>>res,List<Integer>sub,int[]nums,int ind){
        res.add(new ArrayList<>(sub));
        for(int i=ind;i<nums.length;++i){
            sub.add(nums[i]);
            dfs78(res,sub,nums,i+1);
            sub.remove(sub.size()-1);
        }
    }
    public List<List<Integer>> subsetsDFS(int[] nums) {
        List<List<Integer>>res=new ArrayList<>();
        List<Integer>sub=new ArrayList<>();
        dfs78(res,sub,nums,0);
        return res;
    }

    //products of subsets of primes
    public List<Integer> subsetsPrime(int[]nums){
        List<Integer> ans = new ArrayList<>();
        if(nums==null||nums.length==0)
            return ans;
        dfs(ans,nums,1,0);
        return ans;
    }

    public void dfs(List<Integer>ans,int[]nums,int product, int index){
        if(product!=1)
            ans.add(product);
        for(int i=index;i<nums.length;++i){

//            if(i!=index && nums[i]==nums[i-1])
//                continue; //have duplicate
            product *= nums[i];
            dfs(ans,nums,product,i+1);
            product/=nums[i];
        }
    }

    //have duplicate
    //sort first
    //add comment

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>>res=new ArrayList<>();
        res.add(new ArrayList<>());
        int n= nums.length;
        int start=0,newStart=0;
        Arrays.sort(nums);
        for(int i=0;i<n;++i){
            start=(i>0 && nums[i]==nums[i-1])?newStart:0;
            newStart=res.size();
            for(int j=start;j<newStart;++j){
                List<Integer>tmp=new ArrayList<>(res.get(j));
                tmp.add(nums[i]);
                res.add(tmp);
            }
        }
        return res;
    }


    //dfs way
    public void dfs90(List<List<Integer>>res,int[]nums,int index,List<Integer>path){
        res.add(new ArrayList<>(path));
        for(int i=index;i<nums.length;++i){
            if(i>index && nums[i]==nums[i-1])
                continue;
            path.add(nums[i]);
            dfs90(res,nums,i+1,path);
            path.remove(path.size()-1);

        }
    }
    public List<List<Integer>> subsetsWithDupDFS(int[] nums) {
        List<List<Integer>>res=new ArrayList<>();
        Arrays.sort(nums);
        dfs90(res,nums,0,new ArrayList<>());
        return res;
    }


    //79 word search: 不能在原始char[][]里面标记，于是把visited boolean[][]换成stack<position>
    public boolean exist(char[][]board,int x,int y,boolean[][]vis,int ind,String word){
        if(ind==word.length())
            return true;
        if(x<0||x>=board.length||y<0||y>=board[0].length||vis[x][y]||word.charAt(ind)!=board[x][y])
            return false;
        vis[x][y]=true;

        if(exist(board,x+1,y,vis,ind+1,word)||exist(board,x-1,y,vis,ind+1,word)
                ||exist(board,x,y+1,vis,ind+1,word)||exist(board,x,y-1,vis,ind+1,word)){
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


    //word seach II

    class TrieNode {
        public boolean isEnd;
        public StringBuilder word;
        public TrieNode[]children=null;
        public TrieNode(){
            this.isEnd=false;
            children=new TrieNode[26];
            word=new StringBuilder("");
        }
    }
    class Trie {
        public TrieNode root=null;

        public Trie(){
            root=new TrieNode();
        }

        /** Inserts a word into the trie. */
        public void insert(String word) {
            int n=word.length();
            TrieNode cur=root;
            for(int i=0;i<n;++i){
                TrieNode next=cur.children[word.charAt(i)-'a'];
                if(next==null){
                    cur.children[word.charAt(i)-'a']=new TrieNode();
                    next=cur.children[word.charAt(i)-'a'];
                    cur.children[word.charAt(i)-'a'].word.append(cur.word).append(word.charAt(i));
                }

                cur=next;
            }
            cur.isEnd=true;
        }

    }

    public void dfs(TrieNode node,char[][]board,int x,int y,List<String>res){
        if(node!=null && node.isEnd){
            res.add(node.word.toString());
            node.isEnd=false;
        }
        if(node==null)
            return;
        if(x<0||x>=board.length||y<0||y>=board[0].length||board[x][y]=='*')
            return;
        node=node.children[board[x][y]-'a'];
        char c=board[x][y];
        board[x][y]='*';
        dfs(node,board,x+1,y,res);
        dfs(node,board,x-1,y,res);
        dfs(node,board,x,y+1,res);
        dfs(node,board,x,y-1,res);
        board[x][y]=c;
    }
    public List<String> findWords(char[][] board, String[] words) {
        List<String>res=new ArrayList<>();
        Trie t=new Trie();
        for(String word:words){
            t.insert(word);
        }
        if(board.length==0||board[0].length==0)
            return res;
        int m=board.length,n=board[0].length;
        for(int i=0;i<m;++i){
            for(int j=0;j<n;++j){
                dfs(t.root,board,i,j,res);
            }
        }
        return res;
    }
    //merge sorted array
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int k=m+n-1;
        int i=m-1;
        int j=n-1;
        while(i>=0 && j>=0){
            if(nums1[i]>=nums2[j])
                nums1[k--]=nums1[i--];
            else
                nums1[k--]=nums2[j--];
        }
        while(j>=0){
            nums1[k--]=nums2[j--];
        }
    }


    //93 restore IP address
    public boolean valid(String str){
        int n = str.length();
        if(n>1 && str.charAt(0)=='0')
            return false;
        return Integer.parseInt(str)<=255;
    }
    public void dfs(List<String>res,String s,int ind,List<String>path){
        if(ind==s.length() && path.size()==4){
            res.add(String.join(".",path));
            return;
        }

        for(int i=ind+1;i<=s.length()&& i<=ind+3;++i){
            String str = s.substring(ind,i);
            if(valid(str)){
                path.add(str);
                dfs(res,s,i,path);
                path.remove(path.size()-1);
            }
        }
    }

    public List<String> restoreIpAddresses(String s) {
        int n = s.length();
        List<String>res=new ArrayList<>();
        if(n<4||n>12)
            return res;
        dfs(res,s,0,new ArrayList<>());
        return res;
    }

    //124 binary tree maximum path sum
    public int dfs(TreeNode root,int[]res){
        if(root==null)
            return 0;
        int l = dfs(root.left,res);
        int r = dfs(root.right,res);
        int ans = Math.max(root.val,Math.max(l+root.val,r+root.val));
        res[0]=Math.max(res[0],Math.max(ans,root.val+l+r));
        return ans;

    }
    public int maxPathSum(TreeNode root) {
        int[]res={Integer.MIN_VALUE};
        dfs(root,res);
        return res[0];
    }

    //138 copy list with random pointer
    //clone list
    public RandomListNode copyRandomList(RandomListNode head){
        if(head==null)
            return null;
        Map<RandomListNode,RandomListNode>relation = new HashMap<>();
        RandomListNode node = head;
        while(node!=null){
            if(!relation.containsKey(node))
                relation.put(node,new RandomListNode(node.label));

            if(node.next!=null){
                if(!relation.containsKey(node.next))
                    relation.put(node.next,new RandomListNode(node.next.label));
                relation.get(node).next = relation.get(node.next);
            }
            if(node.random!=null){
                if(!relation.containsKey(node.random))
                    relation.put(node.random,new RandomListNode(node.random.label));
                relation.get(node).random = relation.get(node.random);
            }
            //head=head.next;
            node =node.next;
        }
        return relation.get(head);
    }

    //map way
    Map<RandomListNode,RandomListNode>map = new HashMap<>();
    public RandomListNode copyRandomListMap(RandomListNode head) {
        if(head==null)
            return null;
        if(!map.containsKey(head)){
            map.put(head,new RandomListNode(head.label));
            map.get(head).next = copyRandomListMap(head.next);
            map.get(head).random = copyRandomListMap(head.random);
        }
        return map.get(head);
    }

    //143 reorder list
    public ListNode reverseList(ListNode head){
        if(head==null||head.next==null)
            return head;
        ListNode newHead=null;
        while(head!=null){
            ListNode next=head.next;
            head.next=newHead;
            newHead=head;
            head=next;
        }
        return newHead;
    }
    public void reorderList(ListNode head) {
        if(head==null||head.next==null)
            return;
        ListNode fast=head.next;
        ListNode slow=head;
        while(fast!=null && fast.next!=null){
            fast=fast.next.next;
            slow=slow.next;
        }
        fast=slow.next;
        slow.next=null;
        slow=head;
        fast=reverseList(fast);
        while(fast!=null){
            ListNode node1=fast.next;
            ListNode node2=slow.next;
            fast.next=slow.next;
            slow.next=fast;
            fast=node1;
            slow=node2;
        }
    }

    //two dimension array
    /*
    碰到了一个新题，一个二维数组，每一行都只有0和1，前面部分是0，后一部分是1，找到数组里面最左边的1的那一列数。
    用了两种方法，对每一行二分找到第一个1，然后找到整个数组的最小的第一个1的列数。O(mlogn)
    后来说不是最优的，然后改进用O(m+n)的算法，找到第一行第一个1，然后往下找，是0就continue，是1的话往前找.
    最后对比了一下两种方法的优劣，哪种方法在什么情况下比较好 或者哪种情况持续比另一种好。
    直接没结果。。。大概是跪了。。。
     */


    //kth largest element in an array
    public int quickSelect(int[]nums,int begin,int end){

        int low = begin, hi=end,key =nums[low];
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

    //257 binary tree paths
    public void dfs(TreeNode root,List<String>res,String path){
        if(root==null)
            return;
        if(root.left==null && root.right==null){
            res.add(path+root.val);
            return;
        }
        dfs(root.left,res,path+root.val+"->");
        dfs(root.right,res,path+root.val+"->");
    }
    public List<String> binaryTreePaths(TreeNode root) {
        List<String>res=new ArrayList<>();
        dfs(root,res,"");
        return res;
    }

    //3sum smaller 259
    public int threeSumSmaller(int[] nums, int target) {
        int n = nums.length;
        Arrays.sort(nums);
        int res=0;
        for(int i=0;i<=n-2;++i){
            int begin=i+1,end=n-1;
            while(begin<end){
                int sum=nums[i]+nums[begin]+nums[end];
                if(sum>=target){
                    end--;
                }else{
                    res+=end-begin;
                    begin++;
                }
            }
        }
        return res;
    }

    //268 missing number
    public int missingNumber(int[] nums) {
        int ans = 0,n=nums.length;
        for(int i=1;i<=n;++i){
            ans-=nums[i-1];
            ans+=i;
        }
        return ans;
    }

    //Exclusive OR


    //285 inorder successor in bst
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        if(root==null)
            return null;
        if(root.val<=p.val)
            return inorderSuccessor(root.right,p);
        else{
            TreeNode l = inorderSuccessor(root.left,p);
            return l==null?root:l;
        }
    }

    //stack way
    public TreeNode inorderSuccessorIterative(TreeNode root, TreeNode p) {
        if(root==null||p==null)
            return null;
        TreeNode node =root;
        TreeNode succ=null;
        while(node!=null){
            if(node.val<=p.val){
                node=node.right;
            }else{
                succ=node;
                node=node.left;
            }
        }
        return succ;
    }


    /*
    If only nums2 cannot fit in memory, put all elements of nums1 into a HashMap, read chunks of array that fit into the memory,
    and record the intersections.

    If both nums1 and nums2 are so huge that neither fit into the memory,
    sort them individually (external sort), then read 2 elements from each array at a time in memory, record intersections.
     */


    //350 intersection of two arrays II
    public int[] intersect(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int i=0,j=0,m=nums1.length,n=nums2.length;
        List<Integer>ans=new ArrayList<>();
        while(i<m && j<n){
            if(nums1[i]==nums2[j]){
                ans.add(nums1[i]);
                i++;
                j++;
            }else if(nums1[i]>nums2[j])
                j++;
            else
                i++;
        }
        int []res=new int[ans.size()];
        int ind=0;
        for(int x:ans)
            res[ind++]=x;
        return res;
    }


    //393 UTF8 validation
    public boolean validUtf8(int[] data) {
        int count=0;
        for(int d:data){
            d&=0xffff;
            if(count==0){
                if((d>>5)==0b110)
                    count=1;
                else if((d>>4)==0b1110)
                    count=2;
                else if((d>>3)==0b11110)
                    count=3;
                else if((d>>7)==1)
                    return false;
            }else{
                if((d>>6)!=0b10)
                    return false;
                else
                    count--;
            }
        }
        return count==0;
    }


    //415 add strings
    public String addStrings(String num1, String num2) {
        int m = num1.length(), n = num2.length();
        int i=m-1,j=n-1,carry=0;
        StringBuilder sb = new StringBuilder();
        while(i>=0 || j>=0||carry!=0){
            int sum = (i>=0?num1.charAt(i--)-'0':0)+(j>=0?num2.charAt(j--)-'0':0)+carry;
            sb.append(sum%10);
            carry=sum/10;
        }
        sb.reverse();
        return sb.toString();
    }

    //577 total hamming distance
    public int totalHammingDistance(int[] nums) {
        int n = nums.length, res =0;
        for(int i=31;i>=0;--i){
            int ones=0, zeros = 0;
            for(int x:nums){
                if(((x>>i)&0x1)!=0)
                    ones++;
                else
                    zeros++;
            }
            res += ones*zeros;
        }
        return res;
    }


    //525 contiguous array
    //classic
    //fun, have fun
    //should always find a way to binary
    public int findMaxLength(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == 0) nums[i] = -1;
        }

        Map<Integer, Integer> sumToIndex = new HashMap<>();
        sumToIndex.put(0, -1);
        int sum = 0, max = 0;

        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (sumToIndex.containsKey(sum)) {
                max = Math.max(max, i - sumToIndex.get(sum));
            }
            else {
                sumToIndex.put(sum, i);
            }
        }

        return max;
    }

    //543 diameter of binary tree
    private int sum1=1;
    public int dfs543(TreeNode root){
        if(root==null)
            return 0;
        if(root.left==null && root.right==null)
            return 1;
        int l = dfs543(root.left);
        int r =dfs543(root.right);
        sum1=Math.max(sum1,l+r+1);
        return Math.max(l,r)+1;
    }
    public int diameterOfBinaryTree(TreeNode root) {
        dfs543(root);
        return sum1-1;
    }

    //572 subtree of another tree
    public boolean isSameTree(TreeNode p,TreeNode q){
        if(p==null||q==null)
            return p==q;
        return (p.val==q.val) && isSameTree(p.left,q.left) && isSameTree(p.right,q.right);

    }
    public boolean isSubtree(TreeNode s, TreeNode t) {
        if(s==null||t==null)
            return s==t;
        return isSameTree(s,t)||isSubtree(s.left,t)||isSubtree(s.right,t);
    }

    //get preorder and then compare the subtring
    public boolean isSubtreeSubString(TreeNode s, TreeNode t) {
        String spreorder = generatepreorderString(s);
        String tpreorder = generatepreorderString(t);

        return spreorder.contains(tpreorder) ;
    }
    public String generatepreorderString(TreeNode s){
        StringBuilder sb = new StringBuilder();
        Stack<TreeNode> stacktree = new Stack();
        stacktree.push(s);
        while(!stacktree.isEmpty()){
            TreeNode popelem = stacktree.pop();
            if(popelem==null)
                sb.append(",#"); // Appending # inorder to handle same values but not subtree cases
            else
                sb.append(","+popelem.val);
            if(popelem!=null){
                stacktree.push(popelem.right);
                stacktree.push(popelem.left);
            }
        }
        return sb.toString();
    }


    //640 solve the equation
    public int[]convert(String str){
        int coeffic=0,sum=0,sign=1,num=0;
        char[]ss=str.toCharArray();
        int n =ss.length,i=0;
        while(i<n){
            if(ss[i]=='-'){
                if(i>0 && ss[i-1]!='x'){
                    sum+=sign*num;
                    num=0;
                }
                sign=-1;
            }else if(ss[i]=='+'){
                if(i>0 && ss[i-1]!='x'){
                    sum+=sign*num;
                    num=0;
                }
                sign=1;
            }else if(ss[i]=='x'){
                if(i>0 && ss[i-1]=='0' && num==0)
                    coeffic+=0;
                else
                    coeffic+=sign*Math.max(num,1);
                sign=1;
                num=0;
            }else{
                num=10*num+(ss[i]-'0');
            }
            i++;
        }
        if(num!=0)
            sum+=sign*num;
        return new int[]{coeffic,sum};
    }

    public String solveEquation(String equation) {
        String []equations = equation.split("=");
        int []left = convert(equations[0]);
        int []right = convert(equations[1]);
        int coefficient = left[0]-right[0];
        int val = right[1]-left[1];
        if(coefficient==0 && val==0)
            return "Infinite solutions";
        else if(coefficient==0 && val!=0)
            return "No solution";
        else
            return "x="+val/coefficient;
    }


    // Palindromic Substrings
    public int countSubstrings(String s) {
        int n = s.length();
        char[]ss=s.toCharArray();
        boolean [][]dp=new boolean[n+1][n+1];
        dp[0][0]=true;
        for(int i=1;i<=n;++i){
            dp[i][i]=true;
            for(int j=i-1;j>=1;--j){
                if(ss[i-1]==ss[j-1] && (j>i-2||dp[j+1][i-1])){
                    dp[j][i]=true;
                }
            }
        }

        int sum=0;
        for(int i=1;i<=n;++i){
            for(int j=i+1;j<=n;++j){
                if(dp[i][j])
                    sum++;
            }
        }
        return sum+n;
    }


    //654 maximum binary tree
    public TreeNode build(int[]nums,int begin,int end){
        if(begin>end)
            return null;
        if(begin==end)
            return new TreeNode(nums[begin]);
        int val=nums[begin];
        int index=begin;
        int ind=begin+1;
        while(ind<=end){
            if(val<nums[ind]){
                val=nums[ind];
                index=ind;
            }
            ind++;
        }
        TreeNode root = new TreeNode(nums[index]);
        root.left=build(nums,begin,index-1);
        root.right=build(nums,index+1,end);
        return root;
    }
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        int begin=0,end=nums.length-1;
        return build(nums,begin,end);
    }



    //decode ways
    //decode
    public int numDecodings(String s) {
        int n = s.length();
        char []ss=s.toCharArray();
        int[]dp=new int[n+1];
        if(n==0||ss[0]=='0')
            return 0;
        dp[0]=1;
        dp[1]=1;
        for(int i=2;i<=n;++i){
            if(ss[i-1]=='0'){
                if(ss[i-2]=='0'||ss[i-2]>='3')
                    return 0;
                else
                    dp[i]=dp[i-2];
            }else{
                if(ss[i-2]=='1'||(ss[i-2]=='2' && ss[i-1]<='6' && ss[i-1]>='1'))
                    dp[i]=dp[i-2]+dp[i-1];
                else
                    dp[i]=dp[i-1];
            }
        }
        return dp[n];
    }

    //save space
    public int numDecodingsSaveSpace(String s) {
        int n = s.length();
        char []ss=s.toCharArray();
        if(n==0||ss[0]=='0')
            return 0;
        int pre1=1;
        int pre=1;
        int cur=1;
        for(int i=2;i<=n;++i){
            cur=0;
            if(ss[i-1]=='0'){
                if(ss[i-2]=='0'||ss[i-2]>='3')
                    return 0;
                else
                    cur=pre1;
            }else{
                if(ss[i-2]=='1'||(ss[i-2]=='2' && ss[i-1]<='6' && ss[i-1]>='1'))
                    cur=pre+pre1;
                else
                    cur=pre;
            }
            pre1=pre;
            pre=cur;
        }
        return cur;
    }

    public int numDecodingsII(String s) {
        int n = s.length();
        char []ss=s.toCharArray();
        int M = (int)1e9 + 7;
        int[]dp=new int[n+1];
        if(n==0||ss[0]=='0')
            return 0;
        dp[0]=1;
        dp[1]=ss[0]=='*'?9:1;
        for(int i=2;i<=n;++i){
            if(ss[i-1]=='0'){
                if(ss[i-2]=='0'||ss[i-2]>='3')
                    return 0;
                else if(ss[i-2]=='*')
                    dp[i]=2*dp[i-2];
                else
                    dp[i]=dp[i-2];
            }else if(ss[i-1]!='*'){
                if(ss[i-2]!='*'){
                    if(ss[i-2]=='1'||(ss[i-2]=='2' && ss[i-1]<='6' && ss[i-1]>='1'))
                        dp[i]=dp[i-2]+dp[i-1];
                    else
                        dp[i]=dp[i-1];
                }else{
                    dp[i]=dp[i-1];//
                    dp[i] += (ss[i - 1] <= '6') ? (2 * dp[i - 2]) : dp[i - 2];//16~26
                }
            }else{
                //ss[i-1]=='*'
                dp[i] += 9 * dp[i - 1];//* can be anything
                if (ss[i - 2] == '1') dp[i] += 9 * dp[i - 2];//11-19
                else if (ss[i - 2] == '2') dp[i] += 6 * dp[i - 2];//21-26
                else if (ss[i - 2] == '*') dp[i] += 15 * dp[i - 2];//11-19 + 21-26
            }
            dp[i]=dp[i]%M;
        }
        return dp[n];
    }

    //get all possible solutions
    public boolean isValidDouble(String s){
        return s.charAt(0)=='1'||(s.charAt(0)=='2' && s.charAt(1)<='6' && s.charAt(1)>='0');
    }

    public void numDecodeing(List<String>res,String path, String s, int pos){
        if(pos==s.length()){
            res.add(path);
            return;
        }
        if(s.charAt(pos)=='0')
            return;
        int num1 = Integer.parseInt(s.substring(pos,pos+1));
        numDecodeing(res,path+(char)('A'+num1-1),s,pos+1);
        if(s.length()-1>pos && isValidDouble(s.substring(pos,pos+2))){
            int num2 = Integer.parseInt(s.substring(pos,pos+2));
            numDecodeing(res,path+(char)('A'+num2-1),s,pos+2);
        }
    }

    public List<String>numDecodingsAllSolutions(String s){
        List<String>ans = new ArrayList<>();
        if(s.isEmpty()||s.charAt(0)=='0')
            return ans;
        numDecodeing(ans,"",s,0);
        return ans;
    }

    //task scheduler
    public int leastInterval(char[] tasks, int n) {
        if(n==0)
            return tasks.length;
        int []cnt=new int[26];
        int []index=new int[26];
        Arrays.fill(index,-1);
        for(char c:tasks)
            cnt[c-'A']++;
        PriorityQueue<Tuple1>pq=new PriorityQueue<>(new Comparator<Tuple1>(){
            public int compare(Tuple1 t1, Tuple1 t2){
                return t2.y-t1.y;
            }
        });
        for(int i=0;i<26;++i){
            if(cnt[i]!=0)
                pq.offer(new Tuple1((char)(i+'A'),cnt[i]));
        }
        int ans=0,numOfChar=0,nn=tasks.length;
        Queue<Tuple1>q=new LinkedList<>();
        //StringBuilder sb = new StringBuilder();
        while(numOfChar<nn){
            if(!pq.isEmpty()){
                numOfChar++;
                Tuple1 top = pq.poll();
                char cc = top.x;
                index[cc-'A']=ans;
                ans++;
                //sb.append(top.x);
                if(top.y>1)
                    q.offer(new Tuple1(top.x,top.y-1));
                if(!q.isEmpty() && ans-index[q.peek().x-'A']>=n+1)
                    pq.offer(q.poll());
            }else{
                ans++;
                //sb.append('_');
                if(ans-index[q.peek().x-'A']>=n+1)
                    pq.offer(q.poll());
            }
        }
        //System.out.println(sb.toString());
        return ans;
    }


    // arrange missions
    // AABACDCD K=3   A___AB__AC___CD__CD
    public String arrange(String input,int k){
        if(input.length()<=1)
            return input;
        StringBuilder sb = new StringBuilder();
        Map<Character,Integer>missionToTime = new HashMap<>();
        int time = 0;
        char []missions = input.toCharArray();
        for(int i=0;i<missions.length;++i){
            time++;
            if(!missionToTime.containsKey(missions[i])||time-missionToTime.get(missions[i])>k){
               missionToTime.put(missions[i],time);
            }else{
                int gap = k - (time- missionToTime.get(missions[i])-1);
                while(gap>0){
                    sb.append('_');
                    gap--;
                }
                time = k+ missionToTime.get(missions[i])+1;
                missionToTime.put(missions[i],time);
            }
            sb.append(missions[i]);
        }
        return sb.toString();
    }

    //mission order, same task cannot be called in a period

    public int missionOrder(List<Integer>mission, int k){
        if(mission.isEmpty())
            return  0;
        Map<Integer,Integer>map = new HashMap<>();
        int time =0, n = mission.size();
        for(int i=0;i<n;++i){
            time++;
            if(!map.containsKey(mission.get(i))||time-map.get(mission.get(i))>k)
                map.put(mission.get(i),time);
            else{
                time = k + map.get(mission.get(i))+1;
                map.put(mission.get(i),time);
            }
        }
        return time;
    }

    //if cooldown in very small, but there are lots of tasks, how to reduce space
    public int taskScheduleII(int[]tasks,int cooldown){
        if(tasks==null|| tasks.length==0)
            return 0;
        Queue<Integer>q = new LinkedList<>();
        HashMap<Integer,Integer>map = new HashMap<>();
        int slots =0;
        for(int task:tasks){
            if(map.containsKey(task) && map.get(task)>slots){
                slots = map.get(task);
            }
            if(q.size()==cooldown+1){
                map.remove(q.poll());
            }
            map.put(task,slots+cooldown+1);
            q.offer(task);
            slots++;
        }
        return slots;
    }

    //minimize time
    public int taskScheduleIII(int[]tasks,int cooldown){
        Map<Integer,Integer>map = new HashMap<>();
        for(int task: tasks){
            if(!map.containsKey(task)){
                map.put(task,1);
            }else{
                map.put(task,map.get(task)+1);
            }
        }

        int maxFrequencey = 0;
        int countOfMax = 0;
        for(int fre: map.values()){
            if(fre>maxFrequencey){
                maxFrequencey = fre;
                countOfMax=1;
            }else if(maxFrequencey==fre){
                countOfMax++;
            }
        }
        int minimumTime = (maxFrequencey-1)*(cooldown+1)+countOfMax;
        return Math.max(minimumTime,tasks.length);
    }


    public List<Integer> findAnagrams(String haystack, String needle){
        int []cnt = new int[26];
        char []pp= haystack.toCharArray();
        char []ss = needle.toCharArray();
        for(char c:pp)
            cnt[c-'a']++;
        int d = pp.length, start =0 , n = haystack.length(),end=0,m=pp.length;
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

    //move zeros
    //move zero to front, moveone to back, maintain the order of other non-zero and non-one elements
    public void moveZeros(int[]nums){
        if(nums==null||nums.length==0)
            return;
        int insert = 0;
        for(int i=0;i<nums.length;++i){
            if(nums[i]!=1)
                nums[insert++]=nums[i];
        }

        int tmp = insert-1;//save the position before all one
        while(insert<nums.length)
            nums[insert++]=1;
        insert = tmp;
        for(int i=insert;i>=0;--i){
            if(nums[i]!=0)
                nums[insert--]=nums[i];
        }
        while(insert>=0)
            nums[insert--]=0;
    }



//157 read 4
//158 read 4


    public int read4(char[]buf){
        return 4;
    }
    public int read(char[] buf, int n) {
        char []buffer = new char[4];
        int ans = 0;
        while(true){
            int num = read4(buffer);
            for(int i=0;i<num && ans<n;++i)
                buf[ans++]=buffer[i];
            if(ans>=n||num!=4)
                break;
        }
        return Math.min(ans,n);
    }

    public char[]buffer = new char[4];
    public int num=0;
    public int curEnd=0;
    /*
    No, you don't have to take care of this.
The only thing is when you call read4() which reads 4 bytes into your buffer you might read more than you need,
 so you want to store those bytes in the structure,
and next time you call read will start from those stored bytes, then read more from the file


Think that you have 4 chars "a, b, c, d" in the file, and you want to call your function twice like this:

read(buf, 1); // should return 'a'
read(buf, 3); // should return 'b, c, d'
All the 4 chars will be consumed in the first call. So the tricky part of this
question is how can you preserve the remaining 'b, c, d' to the second call
     */
    public int readMultipleTimes(char[] buf, int n) {
        int res=0;
        while(res<n){
            if(curEnd==0)
                num=read4(buffer);
            for(;curEnd<num && res<n;++curEnd)
                buf[res++]=buffer[curEnd];
            if(curEnd==num)
                curEnd=0;
            if(num<4)
                break;
        }
        return Math.min(res,n);
    }



    public double myPow(double x,int n){
        boolean negative = n<0?true:false;
        long nn = Math.abs((long)n);
        if(n==0)
            return 1.0;
        double res =1.0;
        while(nn>0){
            if((nn&0x1)!=0)
                res *=x;
            nn >>=1;
            x *=x;
        }
        return negative?1.0/res:res;
    }

    public double myPowRecursive(double x,int n){
        if(n==0)
            return 1.0;
        if(n<0){
            n= -n ;
            x =1.0/x;
        }
        return n%2==0?myPowRecursive(x*x,n/2):x*myPowRecursive(x*x,n/2);
    }


    //postorder
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer>res=new ArrayList<>();
        TreeNode pre=null;
        TreeNode node=root;
        Stack<TreeNode>stk=new Stack<>();
        while(!stk.isEmpty()||node!=null){
            while(node!=null){
                stk.push(node);
                node=node.left;
            }
            node=stk.peek();
            if(node.right!=null && pre!=node.right){
                node=node.right;
            }else{
                res.add(node.val);
                stk.pop();
                pre=node;
                node=null;
            }

        }
        return res;
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer>res=new ArrayList<>();
        TreeNode node=root;
        Stack<TreeNode>stk=new Stack<>();
        while(node!=null||!stk.isEmpty()){
            while(node!=null){
                stk.push(node);
                node=node.left;
            }
            node=stk.pop();
            res.add(node.val);
            node=node.right;
        }
        return res;
    }


    //230 Kth Smallest Element in a BST
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

    //iterative way
    public int kthSmallestByStack(TreeNode root,int k){
        Stack<TreeNode> stk = new Stack<>();
        while(root!=null||!stk.isEmpty()){
            while(root!=null){
                stk.push(root);
                root= root.left;
            }
            root = stk.pop();
            if(--k ==0)
                break;
            root = root.right;
        }
        return root.val;
    }


    //238 product of array except itself
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[]res=new int[n];
        res[0]=1;
        for(int i=1;i<n;++i)
            res[i]=res[i-1]*nums[i-1];
        int num=1;
        for(int i=n-1;i>=0;--i){
            res[i]*=num;
            num*=nums[i];
        }
        return res;
    }

    public int[] productExceptSelfCostSpace(int[] nums) {
        int n=nums.length;
        int []res=new int[n];//1 a0 a0a1 a0a1a2
        int []after=new int[n];//a1a2a3  a2a3 a3 1
        res[0]=after[n-1]=1;
        for(int i=1;i<n;++i){
            res[i]=res[i-1]*nums[i-1];
            after[n-1-i]=after[n-i]*nums[n-i];
        }
        for(int i=0;i<n;++i)
            res[i]*=after[i];
        return  res;
    }

    //my sqrt
    //69 sqrt(x)
    public int mySqrt(int x) {
        if(x<=1)
            return x;
        int begin = 1, end =x;
        while(begin<end){
            int mid = (end-begin)/2+begin;
            if(mid>46340){
                end = mid;
                continue;
            }
            if(mid<=x/mid && (mid+1)>x/(mid+1))
                return mid;
            else if(mid*mid<x)
                begin=mid+1;
            else
                end =mid;
        }
        return begin;
    }

    //newton
    public int mySqrtNewton(int x) {
        //newton
        if(x<=1)
            return x;
        double first = x*1.0;
        while(Math.abs(first*first-x)>1e-5){
            first=(first+x/first)/2;//(first*first-x)/(2*first);
        }
        return (int)first;
    }


    //274 h-index
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

    //hindex ii
    public int hIndexII(int[] citations) {
        int n=citations.length;
        int end=n,begin=0;
        if(n==0)
            return 0;
        while(begin<end){
            int mid=(end-begin)/2+begin;
            if(citations[mid]>=(n-mid))
                end=mid;
            else
                begin=mid+1;
        };
        return n-begin;
    }

    //cost time
    public int lower(int target,int[]citations){
        //lowerBound
        int n = citations.length;
        if(target>citations[n-1])
            return n;
        int begin=0,end=n-1;
        while(begin<end){
            int mid=(end-begin)/2+begin;
            if(citations[mid]>=target)
                end=mid;
            else
                begin=mid+1;
        }
        return begin;
    }
    public int hIndexMoreTime(int[] citations) {
        //binary search
        int n = citations.length;
        if(n==0)
            return 0;
        if(citations[0]>=n)
            return n;
        int begin =citations[0],end=n;
        while(begin<end){
            int mid =(end-begin)/2+begin;
            int lo = lower(mid,citations);
            if(n-lo>=mid)
                begin=mid+1;
            else
                end=mid;
        }
        return n-lower(begin,citations)>=begin?begin:begin-1;
    }


    //hard part

    //84 Largest Rectangle in Histogram
    public int largestRectangleArea(int[] height) {
        Stack<Integer>stk=new Stack<>();
        int n =height.length,area=0;
        int []heights =new int[n+1];
        for(int i=0;i<n;++i)
            heights[i]=height[i];
        for(int i=0;i<=n;++i){
            while(!stk.isEmpty() && heights[stk.peek()]>heights[i]){
                int h = height[stk.pop()];
                int l = stk.isEmpty()?i:i-stk.peek()-1;
                area=Math.max(area,h*l);
            }
            stk.push(i);
        }
        return area;
    }

    //316 remove duplicate letters
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

    //116 & 117
    //bfs normal way
    public void connect(TreeLinkNode root) {
        TreeLinkNode node =root;
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

    //117 Populating Next Right Pointers in Each Node II
    public void connectII(TreeLinkNode root) {
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


    //398 reservoir sampling
    //random pick index
    public int pick(int target,int[]arrs) {
        Random rand = new Random();
        int n = arrs.length;
        int index =-1, count=1;
        for(int i=0;i<n;++i){
            if(arrs[i]==target){
                if(rand.nextInt(count)==0)
                    index = i;
                count++;
            }
        }
        return index;
    }


    public int findOverlap(Interval []intervals){
        if(intervals==null||intervals.length==0)
            return 0;
        Arrays.sort(intervals, new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return o1.start-o2.start;
            }
        });

        PriorityQueue<Integer>pq = new PriorityQueue<>();
        pq.offer(intervals[0].end);
        int overlapStart = -1;
        int overlapEnd = -1;
        int n = intervals.length;
        for(int i=1;i<n;++i){
            if(intervals[i].start>=pq.peek()){
                pq.poll();
            }else{
                overlapStart = intervals[i].start;
                overlapEnd = Math.min(pq.peek(),intervals[i].end);
            }
            pq.offer(intervals[i].end);
        }
        return overlapStart;
    }

    /*
    interval [startTime, stoptime)   ----integral  time stamps. more info on 1point3acres.com
给这样的一串区间 I1, I2......In
找出 一个 time stamp  出现在interval的次数最多。
startTime <= t< stopTime 代表这个数在区间里面出现过。

example：  [1,3),  [2, 7),   [4,  8),   [5, 9)
5和6各出现了三次， 所以答案返回5，6。
     */

    class Point{
        public int time;
        public boolean isStart;
        public Point(int time,boolean isStart){
            this.time = time;
            this.isStart = isStart;
        }
    }

    public List<Integer>findMaxOverLapTime(List<Interval>intervals){
        List<Integer> ans = new ArrayList<>();
        if(intervals==null||intervals.isEmpty())
            return ans;
        List<Point> points = new ArrayList<>();
        for(Interval interval: intervals){
            points.add(new Point(interval.start,true));
            points.add(new Point(interval.end,false));
        }

        points.sort(new Comparator<Point>() {
            @Override
            public int compare(Point o1, Point o2) {
                return o1.time!=o2.time?o1.time-o2.time:o1.isStart?1:-1;
            }
        });
        int max=0, num=0, start =0, end = 0;
        for(Point point:points){
            if(point.isStart){
                num++;
                if(num>max){
                    max = num;
                    start = point.time;
                    end = point.time;
                }
            }else{
                if(num == max){
                    end = point.time;
                }
                num--;
            }
        }
        for(int i=start;i<end;++i)
            ans.add(i);
        return ans;
    }

    //find intersection of two interval list
    public void findIntersection(Interval[]A, Interval[]B){
        List<Point> points = new ArrayList<>();
        for(Interval interval: A){
            points.add(new Point(interval.start,true));
            points.add(new Point(interval.end,false));
        }
        for(Interval interval: B){
            points.add(new Point(interval.start,true));
            points.add(new Point(interval.end,false));
        }

        points.sort(new Comparator<Point>() {
            @Override
            public int compare(Point o1, Point o2) {
                return o1.time!=o2.time?o1.time-o2.time:o1.isStart?1:-1;
            }
        });

        int  num=0, start =0, end = 0;
        for(Point point:points){
            if(point.isStart){
                num++;
                if(num>1){
                    start = point.time;
                    end = point.time;
                }
            }else{
                if(num >1){
                    end = point.time;
                    System.out.println("["+start+" ,"+end+"]");
                }
                num--;
            }
        }

    }

    class TimeSlot {
        public int time;
        public boolean isStart;
        public TimeSlot(int t, boolean i) {
            time = t;
            isStart = i;
        }
    }

    //return the num of romm used/employee of each non-overlapping time range
    /*
    公司里有好多employee，给出入职和离职的时间段，打印出每个时间段的在职人数
    输入：
    ［1, 2005, 2016］. 1point 3acres
    ［2, 2008, 2014］
    ［3, 2006, 2008］.
    ［4, 2010, 2014］. From 1point 3acres bbs
    输出:
    2005-2006: 1. 1point3acres.com/bbs
    2006-2008: 2
    2008-2010: 2
    2010-2014: 3
    2014-2016: 1
    也是白人小哥，人很nice，第二题各种提醒我。最后我把所有时间排序，不管是入职还是离职时间.
     */
    public List<String> meetingRooms(Interval[] intervals) {
        List<String> res = new ArrayList<>();
        if (intervals == null || intervals.length == 0) {
            return res;
        }
        List<TimeSlot> times = new ArrayList<>();
        for (Interval i : intervals) {//spilt the start time and end time, then sort them
            times.add(new TimeSlot(i.start, true));//use the boolean to regconize if it's a start or end time
            times.add(new TimeSlot(i.end, false));
        }
        Collections.sort(times, new Comparator<TimeSlot>(){
            public int compare(TimeSlot a, TimeSlot b) {
                return a.time - b.time;
            }
        });
        int count = 1;
        int begin = 0;//it's the index of begin time, not the time itself
        for (int i = 1; i < times.size(); i++) {
            if (times.get(i).time != times.get(i - 1).time) {//only add time range to res when there is a diff between two times, if you don't want to 2008-2008, you should add .time
                res.add(times.get(begin).time + "-" + times.get(i).time + ": " + count);//add to res before count is gonna be changed
                begin = i;//update begintime's index
            }
            if (times.get(i).isStart) {//count curr num of people/rooms
                count++;
            }
            else {
                count--;
            }
        }
        return res;
    }

    //amazing number
    /*
    2) there is a better approach:
- for each element we can know for which interval of start index it will count as an amazing number
- so, we get n intervals and need to know what is the best start. start can be between 0 and n
- if we go through 0..n for each interval, when we have a list with all start and all ends, we can
  find the desired lowest start index if interval starts and ends are sorted

2) in detail, assuming the following array:

index: 0, 1, 2, 3, 4, 5, 6,
value: 4, 2, 8, 2, 4, 5, 3,
n = 7

value 4 at index 0: can be used if start index is between 1 and 3
    becaue there must be at least 4 elements before a[0] to satisfy
	a[0] >= index.
	that is 0 + 1 .. n + 0 - a[0]
value 2 at index 1: can be used if start index is between 2 and 6
    that is 1 + 1 .. n + 1 - a[1]
value 8 at index 2 can never be used because 8>n
	that is 2 + 1 .. n + 2 - a[2] => 3 .. 1 (which is not an interval)
value 2 at index 3 can be used if start index is between 4 and 8 (*),
	that is 3 + 1 .. n + 3 - a[3]
value 4 at index 4 can be used if start index is between 5..7
    that is 4 + 1 .. n + 4 - a[4]
value 5 at index 5 can be used if start index is between 6..7
	that is 5 + 1 .. n + 5 - a[5]
value 3 at in dex 6 can be used if start index is between 7..10
	that is 6 + 1 .. n + 6 - a[6]

result: at index 6 (4 values are amazing)
        at index 7 (4 values are amazing)
		note index 7 = 0, 0 < 6, therefore the result is 0
     */
    public static int amazingNumber(int[] a) {
        int len = a.length;
        LinkedList<Interval> intervals = new LinkedList<>();

        // find all the intervals that if the array starts at any index in the interval, there will
        // be at least 1 element is amazing number
        for (int i = 0; i < len; i++) {
            if (a[i] >= len) continue;
            int start = (i+1) % len;
            int end = (len + (i - a[i])) % len;
            System.out.println(i + ": " + start + " - " + end);
            intervals.add(new Interval(start, end));
        }

        // now our problem has just become: "find the year that has the maximum number of people
        // alive"
        int[] count = new int[len];
        for (Interval i : intervals) {
            count[i.start]++;
            if (i.end+1 < count.length) count[i.end+1]--;
        }
        int max = 0;
        int counter = 0;
        int idx = 0;
        for (int i = 0; i < count.length; i++) {
            counter += count[i];
            if (counter > max) {
                max = counter;
                idx = i;
            }
        }

        return idx;
    }

    public int longestConsecutive(int[] nums) {
        Set<Integer>set=new HashSet<>();
        for(int x:nums)
            set.add(x);
        int maxLength=0;
        for(int x:nums){
            if(set.contains(x)){
                set.remove(x);
                int l=x-1;
                int r =x+1;
                while(set.contains(l)){
                    set.remove(l--);
                }
                while(set.contains(r)){
                    set.remove(r++);
                }
                maxLength=Math.max(maxLength,r-l-1);
            }
        }
        return maxLength;
    }


    //276 paint fence
    public int numWays(int n, int k) {
        if(n<=0)
            return 0;
        if(n==1)
            return k;
        int diff = k*(k-1);
        int same = k;
        while(n-- >2){
            int save = diff;
            diff = (diff+same)*(k-1);
            same = save;
        }
        return diff+same;
    }


    //find max sum
    //matrix sum
    //submatrix

    public int maxSubarray(int[]nums){
        int maxSum  = Integer.MIN_VALUE, sum=0;
        for(int x:nums){
            sum+=x;
            if(sum>maxSum)
                maxSum = sum;
            if(sum<0)
                sum=0;
        }
        return maxSum;

    }

    //560 subarray sum equals k
    public int subarraySum(int[] nums, int k) {
        Map<Integer,Integer> map = new HashMap<>();
        int sum =0,n=nums.length,cnt=0;
        map.put(0,1);//哨兵
        for(int i=0;i<n;++i){
            sum+=nums[i];
            int val = map.getOrDefault(sum-k,0);
            cnt+=val;
            map.put(sum,map.getOrDefault(sum,0)+1);
        }
        return cnt;
    }


    //subarray sum
    //handle the negative numbers;
    public boolean subArraySum(int k, int[]nums){
        int sum = 0;
        HashSet<Integer>set = new HashSet<>();
        for(int x:nums){
            sum +=x;
            if(set.contains(sum-k))
                return true;
            set.add(sum);
        }
        return false;
    }

    //all positive number
    public boolean subArraySum(int[]nums,int k){
        if(k<1||nums==null)
            return false;
        int begin =0, end =0, n=nums.length, sum=0;
        while(end<n){
            sum +=nums[end++];
            while(sum>k){
                sum -= nums[begin++];
            }
            if(sum==k)
                return true;
        }
        return false;
    }

    public int maxSumSubmatrix(int[][] matrix, int k) {
        if(matrix.length==0||matrix[0].length==0)
            return 0;
        int m = matrix.length,n=matrix[0].length;
        int maxSum=Integer.MIN_VALUE;
        for(int l=0;l<n;++l){
            int dp[]=new int[m];
            for(int r=l;r<n;++r){
                for(int i=0;i<m;++i)
                    dp[i]+=matrix[i][r];

                //find the largest maxvalue
                maxSum=Math.max(maxSum,maxSubarray(dp));

            }
        }
        return maxSum;
    }

    //also you can use rows
    public boolean subMatrixSum(int[][]nums,int kk){
        if(nums==null||nums.length==0||nums[0]==null||nums[0].length==0)
            return false;
        int m = nums.length;
        int n = nums[0].length;
        for(int i=0;i<m;++i){
            int []row = new int[n];
            for(int j=i;j<m;++j){
                for(int k=0;k<n;++k)
                    row[k]+=nums[j][k];
                if(subArraySum(kk,row))
                    return true;
            }
        }
        return false;
    }

    //209 minimum size subarray sum
    public int minSubArrayLen(int s, int[] nums) {
        //two pointers
        int begin=0,end=0;
        int sum=0,len=Integer.MAX_VALUE;
        int n=nums.length;
        while(end<n){
            sum+=nums[end++];
            while(sum>=s){
                len=Math.min(len,end-begin);
                sum-=nums[begin++];
            }
        }
        return len==Integer.MAX_VALUE?0:len;
    }

    //325 maximum size subarray sum equals k
    public int maxSubArrayLen(int[] nums, int k) {
        int maxLen=0;
        Map<Integer,Integer>map=new HashMap<>();
        int sum=0,n=nums.length;
        for(int i=0;i<n;++i){
            sum+=nums[i];
            if(sum==k)
                maxLen=Math.max(maxLen,i+1);
            if(map.containsKey(sum-k)){
                maxLen=Math.max(maxLen,i-map.get(sum-k));
            }
            if(!map.containsKey(sum))
                map.put(sum,i);
        }
        return maxLen;
    }

    //410 split array largest sum
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

    //68 text justication
    public List<String> fullJustify(String[] words, int maxWidth) {
        int n=words.length;
        List<String>res = new ArrayList<>();
        if(n==0||maxWidth==0){
            res.add("");
            return res;
        }
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

    //234 palindrome linked list

    public boolean isPalindrome(ListNode head) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode fast = head;
        ListNode slow = dummy;
        while(fast!=null && fast.next!=null){
            fast = fast.next.next;
            slow= slow.next;
        }
        //System.out.println(slow.val);
        //System.out.println(slow.next.val);
        ListNode second= reverseList(slow.next);
        slow.next=null;
        ListNode first = head;
        while(first!=null && second!=null){
            if(first.val!=second.val)
                return false;
            first = first.next;
            second = second.next;
        }
        return true;
    }

    //255 verify preorder
    public boolean verifyPreorder(int[] preorder) {
        //save space
        int minVal=Integer.MIN_VALUE;
        int n = preorder.length;
        int ind=-1;
        for(int i=0;i<n;++i){
            if(preorder[i]<minVal)
                return false;
            while(ind>-1 && preorder[ind]<=preorder[i]){
                minVal=preorder[ind--];
            }
            preorder[++ind]=preorder[i];
        }
        return true;
    }

    //stack way
    public boolean verifyPreorderStack(int[] preorder) {
        //stack
        Stack<Integer>stk=new Stack<>();
        int minVal=Integer.MIN_VALUE;
        int n = preorder.length;
        for(int i=0;i<n;++i){
            if(preorder[i]<minVal)
                return false;
            while(!stk.isEmpty() && preorder[stk.peek()]<=preorder[i]){
                minVal=preorder[stk.pop()];
            }
            stk.push(i);
        }
        return true;
    }


    ///331 verify preorder serialize of a binary tree
    public boolean isValidSerialization(String preorder) {
        int diff=1;
        //not null node, 1 in,2 out
        //null node, 1 in ,0 out;
        String []args=preorder.split(",");
        for(String str:args){
            diff--;
            if(diff<0)
                return false;
            if(!str.equals("#"))
                diff+=2;
        }
        return diff==0;
    }


    //cost time
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
    public boolean isValidSerializationCostTime(String preorder) {
        Stack<String>stk=new Stack<>();
        //System.out.println(preorder);
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
        //System.out.println(stk.size());
        return stk.size()==1 && stk.peek().equals("#");

    }


    //lcs longest common subsequence
    //dp[i][j] = s1[i-1]==s2[j-1]?dp[i-1][j-1]+1:max(dp[i-1][j],dp[i][j-1])

    // has common substring
    public boolean hasCommonThanK(String A, String B, int k){
        if(k<1)
            return true;
        int m = A.length(), n = B.length();
        int [][]dp = new int[m+1][n+1];
        for(int indexA =1; indexA<=m;++indexA){
            for(int indexB =1; indexB<=n;++indexB){
                if(A.charAt(indexA-1)==B.charAt(indexB-1)){
                    dp[indexA][indexB] = dp[indexA-1][indexB-1]+1;
                }
                if(dp[indexA][indexB]>=k){
                    System.out.println(A.substring(indexA-dp[indexA][indexB],indexA));
                    System.out.println(B.substring(indexB-dp[indexA][indexB],indexB));
                    return true;
                }
            }
        }
        return false;
    }

    //394 decode string
    //recursive way
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

    //stack way
    public String decodeStringStack(String s) {
        int ind=0,n = s.length();
        char []ss = s.toCharArray();
        Stack<Integer>num=new Stack<>();
        Stack<String>stk=new Stack<>();
        StringBuilder sb = new StringBuilder();
        while(ind<n){
            if(Character.isDigit(ss[ind])){
                int number = 0;
                while(ind<n && Character.isDigit(ss[ind])){
                    number=10*number+(ss[ind++]-'0');
                }
                num.push(number);
            }
            if(ind<n && ss[ind]=='['){
                stk.push(sb.toString());
                sb.setLength(0);
                ind++;
            }else if(ind<n && ss[ind]==']'){
                StringBuilder tmp = new StringBuilder(stk.pop());
                int repeat = num.pop();
                while(repeat-- >0){
                    tmp.append(sb.toString());
                }
                sb=tmp;
                ind++;
            }else if(ind<n)
                sb.append(ss[ind++]);

        }
        return sb.toString();
    }


    //239 Sliding Window Maximum
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



    //sliding window minimum
    public int[]minSlidingWindow(int[]nums,int k){
        Deque<Integer>dq = new LinkedList<>();
        int n = nums.length;
        int []ans = new int[n-k+1];
        if(n==0||k==0||n<k)
            return ans;
        int ind=0;
        for(int i=0;i<n;++i){
            if(!dq.isEmpty() && dq.peekFirst()<i-k+1)
                dq.pollFirst();
            while(!dq.isEmpty() && nums[dq.peekLast()]<nums[i])
                dq.pollLast();
            dq.push(i);
            if(i>=k-1)
                ans[ind++]=nums[dq.peekFirst()];
        }
        return ans;
    }

    //binary tree ancestor of deepest leaf
    class ReturnVal{
        public int depth;
        public TreeNode node;
        public ReturnVal(int depth, TreeNode node){
            this.depth = depth;
            this.node = node;
        }

        public ReturnVal(){
            this.depth = 0;
            this.node = null;
        }
    }

    public ReturnVal findLca(TreeNode root){
        if(root==null)
            return new ReturnVal();
        ReturnVal l = findLca(root.left);
        ReturnVal r = findLca(root.right);
        ReturnVal res = new ReturnVal();

        if(l.depth==r.depth){
            res.node = root;
            res.depth = l.depth+1;
        }else if(l.depth>r.depth){
            res.node = l.node;
            res.depth = l.depth+1;
        }else{
            res.node = r.node;
            res.depth = r.depth+1;
        }
        return res;
    }

    public TreeNode findLcaRecursive(TreeNode node){
        return findLca(node).node;
    }


    //iterative way
    //bfs find the two end of last level, use parent map to find the relation between child and parent
    public TreeNode findLcaIterative(TreeNode node){
        if(node==null)
            return null;
        Map<TreeNode,TreeNode>parent = new HashMap<>();
        Queue<TreeNode>q=new LinkedList<>();
        q.offer(node);
        TreeNode left = null;
        TreeNode right = null;
        while(!q.isEmpty()){
            int size = q.size();
            int save = size;
            while(size-- >0){
                TreeNode cur = q.poll();
                if(size==save-1)
                    left = cur;
                if(size==0)
                    right= cur;
                if(node.left!=null){
                    q.offer(node.left);
                    parent.put(node.left,node);
                }
                if(node.right!=null){
                    q.offer(node.right);
                    parent.put(node.right,node);
                }
            }
        }

        while(left!=right){
            left = parent.get(left);
            right = parent.get(right);
        }
        return left;
    }


    //contains duplicate
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer>set=new HashSet<>();
        int n = nums.length;
        for(int i=0;i<n;++i){
            if(set.contains(nums[i]))
                return true;
            set.add(nums[i]);
            if(i>=k)
                set.remove(nums[i-k]);
        }
        return false;
    }


    //413 && 446
    //arithmetic slices
    //arithmetic slicess II -subsequence
    public int numberOfArithmeticSlices(int[] A) {
        int n = A.length;
        if(n<3)
            return 0;
        int cur=0,sum=0;
        for(int i=2;i<n;++i){
            if(A[i]-A[i-1]==A[i-1]-A[i-2]){
                cur++;
            }else
                cur=0;
            sum+=cur;
        }
        return sum;
    }


    public int numberOfArithmeticSlicesII(int[] A) {
        int res = 0;
        Map<Integer, Integer>[] map = new Map[A.length];

        for (int i = 0; i < A.length; i++) {
            map[i] = new HashMap<>(i);

            for (int j = 0; j < i; j++) {
                long diff = (long)A[i] - A[j];
                if (diff <= Integer.MIN_VALUE || diff > Integer.MAX_VALUE) continue;

                int d = (int)diff;
                int c1 = map[i].getOrDefault(d, 0);
                int c2 = map[j].getOrDefault(d, 0);
                res += c2;
                map[i].put(d, c1 + c2 + 1);
            }
        }

        return res;
    }


    //find the longest arithmetic sequence
    public int findLongest(List<Integer>input){
        int n = input.size();
        if(n<=2)
            return n;
        int maxLen = 0;
        int[][]dp = new int[n][n];
        Map<Integer,List<Integer>>valueToIndex = new HashMap<>();
        for(int i=0;i<n;++i){
            if(!valueToIndex.containsKey(input.get(i)))
                valueToIndex.put(input.get(i),new ArrayList<>());
            valueToIndex.get(input.get(i)).add(i);
        }
        for(int index =1;index<n;++index){
            for(int secondLast = index-1;secondLast>=0;--secondLast){
                int gap = input.get(index)-input.get(secondLast);
                int next = input.get(secondLast)-gap;
                if(valueToIndex.containsKey(next)){
                    int nextIndex = -1;
                    for(int j= valueToIndex.get(next).size()-1;j>=0;--j){
                        if(valueToIndex.get(next).get(j)<secondLast){
                            nextIndex = valueToIndex.get(next).get(j);
                            break;
                        }
                    }
                    if(nextIndex!=-1){
                        dp[secondLast][index] = dp[nextIndex][secondLast]+1;
                        maxLen = Math.max(maxLen,dp[secondLast][index]);
                    }
                }
                if(dp[secondLast][index]==0)
                    dp[secondLast][index]=2;
            }
        }
        return maxLen;
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


    //360 sort transform array
    public int getValue(int a,int b,int c,int x){
        return a*x*x+b*x+c;
    }
    public int[] sortTransformedArray(int[] nums, int a, int b, int c) {
        int i=0,n=nums.length,j=n-1;
        int ind=0,revInd=n-1;
        int []res=new int[n];
        while(i<=j){
            int iVal=getValue(a,b,c,nums[i]);
            int jVal=getValue(a,b,c,nums[j]);
            if(a>=0){
                if(iVal>=jVal){
                    res[revInd--]=iVal;
                    i++;
                }else{
                    res[revInd--]=jVal;
                    j--;
                }
            }else{
                if(iVal<=jVal){
                    res[ind++]=iVal;
                    i++;
                }else{
                    res[ind++]=jVal;
                    j--;
                }
            }
        }
        return res;
    }

    // [-2,-1,1,2]
    //sort array based on their abs value
    public int [] sortedSquares(int []arr){
        if(arr==null||arr.length<=1)
            return arr;
        int n = arr.length;
        int []ans = new int[n];
        int right=0;
        //find the cross point
        while(right<n && arr[right]<0)
            right++;
        int left = right-1;
        for(int i=0;i<n;++i){
            int leftVal = left>=0?Math.abs(arr[left]):Integer.MAX_VALUE;
            int rightVal = right<n?Math.abs(arr[right]):Integer.MAX_VALUE;

            if(rightVal<=leftVal){
                ans[i] = arr[right];
                right++;
            }else{
                ans[i]=arr[left];
                left--;
            }
        }
        return ans;
    }

    List<Integer>ress=null;
    public void printLeft(TreeNode root){
        if(root==null || root.left==null && root.right==null)
            return;
        //System.out.println(root.val);
        ress.add(root.val);
        if(root.left!=null)
            printLeft(root.left);
        else if(root.right!=null)
            printLeft(root.right);
    }

    public void printRight(TreeNode root){
        if(root==null || root.left==null && root.right==null)
            return;
        if(root.right!=null)
            printRight(root.right);
        else if(root.left!=null)
            printRight(root.left);
        //System.out.println(root.val);
        ress.add(root.val);
    }

    public void printLeaf(TreeNode root){
        if(root==null)
            return;
        if(root.left==null && root.right==null){
            //System.out.println(root.val);
            ress.add(root.val);
        }
        printLeaf(root.left);
        printLeaf(root.right);
    }


    public List<Integer> boundaryOfBinaryTree(TreeNode root) {
        //System.out.println(root.val);
        ress=new ArrayList<>();
        if(root==null)
            return ress;
        ress.add(root.val);
        printLeft(root.left);
        printLeaf(root.left);
        printLeaf(root.right);
        printRight(root.right);
        return ress;
    }

    //max k product
    public long maxK_Product(Integer[]nums,int k){
        if(nums==null||nums.length==0||k<=0||k>nums.length)
            return 0;
        long result = Long.MIN_VALUE;
        Arrays.sort(nums, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                if((long)o1*(long)o2>0)
                    return o2-o1;
                else if((long)o1*(long)o2==0)
                    return o1==0?1:-1;
                else
                    return o1>0?1:-1;
            }
        });
        long tmp =1;
        int count =0;
        for(int i=0;i<nums.length;++i){
            if(nums[i]==0){
                result = Math.max(0,result);
                break;
            }
            count++;
            tmp *=nums[i];
            if(count==k){
                result = Math.max(result,tmp);
                tmp = tmp /nums[i+1-k];
                count--;
            }
        }
        return result;
    }


    //shortest path between knight and target
    public int knightToTarget(int[][]matrix,int x1,int y1,int x2,int y2){
        //move
        int [][]move ={{1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1}};
        if(matrix==null||matrix.length==0||matrix[0]==null||matrix[0].length==0)
            return -1;
        int m = matrix.length, n= matrix[0].length;
        if(!validCoordinate(m,n,x1,y1)||!validCoordinate(m,n,x2,y2))
            return -1;
        Queue<Integer>q = new LinkedList<>();
        boolean [][]vis = new boolean[m][n];
        int res = 0;
        q.offer(x1*n+y1);
        vis[x1][y1]=true;
        while(!q.isEmpty()){
            int size = q.size();
            while(size -- >0){
                int key = q.poll();
                int x = key/n;
                int y = key%n;
                if(x==x2 && y==y2)
                    return res;
                for(int k=0;k<8;++k){
                    int nextX = x+move[k][0];
                    int nextY = y+ move[k][1];
                    if(validCoordinate(m,n,nextX,nextY) && !vis[nextX][nextY]){
                        q.offer(nextX*n+nextY);
                        vis[nextX][nextY]=true;
                    }

                }
            }
            res++;
        }
        return -1;
    }

    public boolean validCoordinate(int m,int n, int i, int j){
        if(i<0||i>=m||j<0||j>=n)
            return false;
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





    class UnionFind{
        public int[]parent=null;
        public int[]rank =null;
        public UnionFind(int n){
            parent = new int[n+1];
            rank=new int[n+1];
            for(int i=0;i<=n;++i)
                parent[i]=i;
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

    public boolean validTree(int n, int[][] edges) {
        int m=edges.length;
        if(n!=m+1)
            return false;
        UnionFind uf = new UnionFind(n);
        for(int[]edge:edges){
            if(uf.mix(edge[0],edge[1])){
                n--;
            }
        }
        return n==1;
    }

    //minimum cover interval
    public int findCover(Interval []intervals, Interval interval){
        Arrays.sort(intervals, new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                return o1.start!=o2.start?o1.start-o2.start:o1.end-o2.end;
            }
        });
        int count = 0;
        int start = interval.start;
        int end = -1;
        int index = 0;
        while (index < intervals.length && end < interval.end) {
            if (intervals[index].end <= start) {
                index++;
                continue;
            }
            if (intervals[index].start > start) {
                break;
            }
            while (index < intervals.length && end < interval.end && intervals[index].start <= start) {
                end = Math.max(intervals[index].end, end);
                index++;
            }
            if (start != end) {
                count++;
                start = end;
            }
        }
        if (end < interval.end) {
            return 0;
        }
        return count;
    }


    public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        if (newInterval == null) {
            return intervals;
        }
        List<Interval> res = new ArrayList<>();
        for (int i = 0; i < intervals.size(); i++) {
            if (intervals.get(i).start <= newInterval.end && intervals.get(i).end >= newInterval.start) {//merge overlaps
                newInterval.start = Math.min(newInterval.start, intervals.get(i).start);//remember to update the start too!!!
                newInterval.end = Math.max(newInterval.end, intervals.get(i).end);
            }
            else {
                res.add(intervals.get(i));//add all non-overlapping intervals
            }
        }
        res.add(newInterval);//add the newInterval when all intervals have been checked so that no overlap exists
        return res;
    }
}
