import commons.ListNode;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by tao on 10/17/17.
 */

class TrieNode{
    public boolean isEnd;
    public TrieNode[]children;
    public TrieNode(){
        isEnd=false;
        children = new TrieNode[26];
    }
}

class Trie{
    private TrieNode root;

    public Trie(){
        root = new TrieNode();
    }

    public void insert(String word){
        TrieNode p = root;
        char []ss = word.toCharArray();
        for(char cc:ss){
            if(p.children[cc-'a']==null)
                p.children[cc-'a']=new TrieNode();
            p=p.children[cc-'a'];
        }
        p.isEnd=true;
    }

    public boolean search(String word){
        TrieNode p = root;
        char []ss = word.toCharArray();
        for(char cc:ss){
            if(p.children[cc-'a']==null)
                return false;
            p=p.children[cc-'a'];
        }
        return p.isEnd;
    }


    public void dfs(TrieNode p, List<String>ans,String prefix){
        if(p==null)
            return;
        if(p.isEnd){
            ans.add(prefix);
           // return;
        }
        for(int i=0;i<26;++i){
            if(p.children[i]!=null)
                dfs(p.children[i],ans,prefix+(char)(i+'a'));
        }
    }
    public List<String>getAllWords(String prefix){
        List<String>ans = new ArrayList<>();
        char []ss = prefix.toCharArray();
        TrieNode p = root;
        for(char cc:ss){
            p = p.children[cc-'a'];
            if(p==null)
                return ans;
        }
        dfs(p,ans,prefix);
        return ans;
    }
}
public class ExpediaOnsite {


    //is palindrome list
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
        //System.out.println(newHead.val);
        return newHead;
    }
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

    //auto completion
    //you can use trie
    //deck of cards

    //if node is null or node is tail
    public boolean deleteNode(ListNode node){
        if(node==null||node.next==null)
            return false;
        node.val = node.next.val;
        node.next = node.next.next;
        return true;
    }


    //string compress
    // oo design, design pattern
    public String compress(String str){
        StringBuilder sb = new StringBuilder();
        int n = str.length();
        int cnt=1,ind=1;
        //pay attention to this situation
        while(ind<=n){
            if(ind<n && str.charAt(ind)==str.charAt(ind-1))
                cnt++;
            else{
                sb.append(str.charAt(ind-1)).append(cnt);
                cnt=1;
            }
            ind++;
        }
        return sb.length()>str.length()?str:sb.toString();
    }


    //19 question
    //skyline problem
    public ListNode removeNthNode(ListNode head,int n){
        ListNode first = head;
        if(n<=0)
            return head;
        while(first!=null && n>0){
            first = first.next;
            n--;
        }
        if(n>0)
            return head;
        ListNode dummy  = new ListNode(0);
        dummy.next = head;
        ListNode p = dummy;
        while(first!=null){
            p = p.next;
            first = first.next;
        }

        p.next = p.next.next;
        return dummy.next;
    }


    //ashley loves numebr
    public int notRepeat(int n){
        int []bits = {0,0,0,0,0,0,0,0,0,0};
        while(n!=0){
            int digit = n%10;
            n/=10;
            if(bits[digit]!=0)
                return 0;
            else
                bits[digit]++;
        }
        return 1;
    }
    public int getSpecialNumber(int m,int n){
        //preprocess
        int []num = new int[1000001];
        for(int i=1;i<=1000000;++i)
            num[i]=num[i-1]+notRepeat(i);
        return num[m]-num[n-1];
    }


    //counting groups, you can find in the expedia

    // 给一个字符串，求出所有可能的组合方式，例如input 为 "ab"， output 为" ["a","b","ab","ba"]"


    //behavioral question
    // 如果你的project 不work该怎么办，



    //longest repeated string: hard question

    // 问了我现在工作做些啥，有没有问题问她，没有就送我出来了。


    //how many triangle can form in an array;

    public void dfs(int[]res,int[]nums,int ind,List<Integer>path){
        if(path.size()==3){
            if(path.get(0)+path.get(1)>path.get(2) && path.get(0)+path.get(2)>path.get(1) && path.get(2)+path.get(1)>path.get(0))
                res[0]++;
            System.out.println(path);
            return;
        }
        for(int i=ind;i<nums.length;++i){
            path.add(nums[i]);
            dfs(res,nums,i+1,path);
            path.remove(path.size()-1);
        }
    }

    public int numberOfTriangle(int[]nums){
        //C(N,3)
        int []res={0};
        dfs(res,nums,0,new ArrayList<>());
        return res[0];
    }


    //设计： 前端有0.3%的请求timeout，后端是15个数据库，这些timeout可能是什么导致的，怎么解决的
    //html code: 404 not found , 200 ok Standard response for successful HTTP requests, 500 Internal Server Error, A generic error message, given when an unexpected condition was encountered and no more specific message is suitable
    //java 垃圾回收好处
    //finally block里面写啥
    //constant 和immutable的区别


    //给你个字符串，要压缩字符串，只有xyz构成，遇到相同的两个，比如xx，压缩成写x，遇到不同的两个，比如xy，就压缩成第三个，z。
    /*
    压缩到不能压缩为止，比如xxx压到写，xyz压倒z， xyyz变成y；
     */


    //Beautiful Arrangement
    public int dfs(int end,int num,boolean[]vis,int start){
        if(num==end+1)
            return 1;
        int res=0;
        for(int i=1;i<=end;++i){
            if(!vis[i] && (i%num==0||num%i==0)){
                vis[i]=true;
                res+=dfs(end,num+1,vis,i);
                vis[i]=false;
            }
        }
        return res;
    }
    public int countArrangement(int N) {
        boolean []vis=new boolean[N+1];
        return  dfs(N,1,vis,0);
    }

    //按照reference 传递有啥坏处
    /*
    unit test integration test是啥
    接口和抽象类的区别
    java 7 & 8的区别
    agile的环境用过没
    什么样的code是好code
    什么样的注释是好注释
    什么事pesudo code？ 有什么好处
    什么是递归，递归的坏处

     cc150的题目


     扫雷的ood
     如何解决一些已有的系统服务的问题，如何提高performance, 如何解决peak time的高峰处理，如何解决服务器不够用的问题，
     如何做一个cloud的build server， 可以从git中拿到source code，然后build 好传给user，说下build server中发生什么




     ood:

     you should inquire who is going to use it and how they are going to use it.who what,where when how why

     for example, suppose you were asked to described the ood oriented design for a coffee maker.

     your coffee maker might be an industrial machine designed to be used in a massive restaurant servicing
     hundreds of customers per hour and making ten different kinds of coffee products, or it might be a very
     simple machine, designed to be used by the elderly for just simple black coffee, these use cases will
     significantly impact your design



     2. define the core object

     3. analyze relationships: which object are members of which other objects? do any objects inherit from any
     others? are relationship many-to-many or one-to-many


     4. investigate actions

     把cc150ood做完了就可以了


     recursion: is a method where the solution to a problem depends on solutions to smaller instances of same problem



     */

}
